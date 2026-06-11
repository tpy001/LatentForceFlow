import collections
import dataclasses
import logging
import math
import pathlib
import time
import typing

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import websockets.exceptions as _websockets_exceptions

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10
    server_retry_interval: float = 5.0
    server_wait_timeout: float = 0.0

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"
        # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_failure_videos: bool = True
    save_success_videos: bool = False

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _create_policy_client(args)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            # action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # if not action_plan:
                    if True:
                        # Finished executing previous action chunk -- compute new chunk
                        obs = env.env._get_observations(force_update=True)

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save both camera views for replay video: agentview on the left, wrist view on the right.
                        replay_images.append(np.concatenate([img, wrist_img], axis=1))

                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_result, client = _infer_with_server_retry(client, element, args)
                        action_chunk = action_result["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            if (done and args.save_success_videos) or (not done and args.save_failure_videos):
                video_path = _save_rollout_video(
                    replay_images=replay_images,
                    video_out_path=pathlib.Path(args.video_out_path),
                    task_suite_name=args.task_suite_name,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    task_description=task_description,
                    success=done,
                )
                if video_path is not None:
                    logging.info(f"Saved rollout video: {video_path}")

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _save_rollout_video(
    *,
    replay_images: typing.Sequence[np.ndarray],
    video_out_path: pathlib.Path,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    task_description: str,
    success: bool,
) -> typing.Optional[pathlib.Path]:
    if not replay_images:
        logging.warning("No replay frames collected; skipping rollout video.")
        return None

    status = "success" if success else "failure"
    task_segment = _sanitize_path_segment(task_description, max_length=80)
    output_dir = video_out_path / task_suite_name / status
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"task_{task_id:03d}_episode_{episode_idx:03d}_{task_segment}.mp4"
    imageio.mimwrite(
        video_path,
        [np.asarray(x) for x in replay_images],
        fps=10,
    )
    return video_path


def _sanitize_path_segment(value: str, *, max_length: int) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    sanitized = "_".join(segment for segment in sanitized.split("_") if segment)
    if not sanitized:
        return "rollout"
    return sanitized[:max_length]


def _create_policy_client(args: Args):
    return _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)


def _infer_with_server_retry(client, element, args: Args):
    start_time = time.monotonic()
    while True:
        try:
            return client.infer(element), client
        except Exception as e:
            if not _is_policy_connection_error(e):
                raise
            if args.server_wait_timeout > 0 and time.monotonic() - start_time >= args.server_wait_timeout:
                raise
            logging.warning(
                "Policy server request failed (%s). Waiting %.1fs for %s:%s before retrying.",
                e,
                args.server_retry_interval,
                args.host,
                args.port,
            )
            time.sleep(args.server_retry_interval)
            client = _create_policy_client(args)


def _is_policy_connection_error(error: Exception) -> bool:
    return isinstance(
        error,
        (
            ConnectionError,
            OSError,
            TimeoutError,
            _websockets_exceptions.ConnectionClosed,
        ),
    )


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
