import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeedbackTrainingChunk:
    """Training payload produced from a feedback window inside `train_interactive_learning_repetition`."""
    obs_proc: Any = None
    negative_action_ta: Optional[np.ndarray] = None
    optimal_action_ta: Optional[np.ndarray] = None
    h_ta: Optional[np.ndarray] = None
    intervention_signal_ta: bool = False
    window_lengths: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    signal_snapshot: tuple[bool, ...] = field(default_factory=tuple)

    def has_data(self) -> bool:
        return self.obs_proc is not None

    def feedback_count_delta(self) -> int:
        return int(self.intervention_signal_ta)

    def training_last_action(self, agent_type: str) -> Optional[np.ndarray]:
        return self.negative_action_ta if agent_type != "Diffusion" else self.optimal_action_ta

    def log_training_chunk(self):
        if not self.has_data():
            return

        logger.debug('intervention_signal:  %s', list(self.signal_snapshot))
        if self.intervention_signal_ta:
            obs_len, signal_len, negative_len, positive_len, optimal_len = self.window_lengths
            logger.info('-----------Add one feedback, len(obs_list):  %s', obs_len)
            logger.debug('len(intervention_signal):  %s  len(action_negative_list):  %s  len +: %s  len *:  %s', signal_len, negative_len, positive_len, optimal_len)

    def log_success_padding_chunk(self):
        if not self.has_data():
            return
        logger.debug('len obslist:  %s  intervention_signal_Ta:  %s', self.window_lengths[0], self.intervention_signal_ta)


class FeedbackWindowBuffer:
    """Store and slice Ta-length teacher-correction windows for `train_interactive_learning_repetition`."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs_list = []
        self.action_positive_list = []
        self.action_negative_list = []
        self.action_teacher_optimal_list = []
        self.intervention_signal = []

    def latest_intervention_active(self) -> bool:
        return len(self.intervention_signal) > 0 and self.intervention_signal[-1] is True

    def ready(self, action_horizon: int) -> bool:
        return len(self.obs_list) >= action_horizon

    def can_flush_success_padding(self) -> bool:
        return len(self.obs_list) > 1

    def append_step(
        self,
        receive_feedback_phase: bool,
        obs_proc,
        h,
        h_no_threshold,
        last_action,
        teacher_action_i,
        ta_i_teacher: int,
    ):
        """Append one step of feedback state, or clear the window when feedback stops."""
        if not receive_feedback_phase:
            self.reset()
            return h, ta_i_teacher

        self.obs_list.append(obs_proc)
        if h is not None and np.any(h):
            self.action_positive_list.append(h)
            self.intervention_signal.append(True)
        elif self.latest_intervention_active():
            h = h_no_threshold
            self.action_positive_list.append(h)
            self.intervention_signal.append(True)
        else:
            ta_i_teacher = 0
            self.action_positive_list.append(h_no_threshold)
            self.intervention_signal.append(False)

        self.action_negative_list.append(last_action)
        self.action_teacher_optimal_list.append(teacher_action_i)
        return h, ta_i_teacher

    def pop_training_chunk(self, action_horizon: int) -> FeedbackTrainingChunk:
        """Return the next Ta-sized training chunk and advance the window by one step."""
        if not self.ready(action_horizon):
            return FeedbackTrainingChunk()

        data_id = 1
        chunk = FeedbackTrainingChunk(
            obs_proc=self.obs_list[data_id],
            intervention_signal_ta=all(
                self.intervention_signal[data_id - 1 : data_id - 1 + action_horizon]
            ),
            window_lengths=self._window_lengths(),
            signal_snapshot=tuple(self.intervention_signal),
        )
        if chunk.intervention_signal_ta:
            chunk.negative_action_ta = self._stack_actions(
                self.action_negative_list[data_id - 1 : data_id - 1 + action_horizon]
            )
            chunk.h_ta = self._stack_actions(
                self.action_positive_list[data_id - 1 : data_id - 1 + action_horizon]
            )
            chunk.optimal_action_ta = self._stack_actions(
                self.action_teacher_optimal_list[data_id - 1 : data_id - 1 + action_horizon]
            )

        self._pop_front()
        return chunk

    def flush_success_padding_chunk(self, action_horizon: int) -> FeedbackTrainingChunk:
        """Return one padded success chunk during episode-end flushing and advance the window."""
        if not self.can_flush_success_padding():
            return FeedbackTrainingChunk()

        data_id = 1
        chunk = FeedbackTrainingChunk(
            obs_proc=self.obs_list[data_id],
            intervention_signal_ta=all(self.intervention_signal[data_id - 1 : -1]),
            window_lengths=self._window_lengths(),
            signal_snapshot=tuple(self.intervention_signal),
        )
        if chunk.intervention_signal_ta:
            negative_actions = self._pad_actions(
                self.action_negative_list[data_id - 1 : data_id - 1 + action_horizon],
                action_horizon,
            )
            positive_actions = self._pad_actions(
                self.action_positive_list[data_id - 1 : data_id - 1 + action_horizon],
                action_horizon,
            )
            optimal_actions = self._pad_actions(
                self.action_teacher_optimal_list[data_id - 1 : data_id - 1 + action_horizon],
                action_horizon,
            )
            chunk.negative_action_ta = self._stack_actions(negative_actions)
            chunk.h_ta = self._stack_actions(positive_actions)
            chunk.optimal_action_ta = self._stack_actions(optimal_actions)

        self._pop_front()
        return chunk

    def _window_lengths(self) -> tuple[int, int, int, int, int]:
        return (
            len(self.obs_list),
            len(self.intervention_signal),
            len(self.action_negative_list),
            len(self.action_positive_list),
            len(self.action_teacher_optimal_list),
        )

    def _pop_front(self):
        self.obs_list.pop(0)
        self.intervention_signal.pop(0)
        self.action_negative_list.pop(0)
        self.action_positive_list.pop(0)
        self.action_teacher_optimal_list.pop(0)

    @staticmethod
    def _pad_actions(actions, action_horizon: int):
        if len(actions) < action_horizon:
            actions = list(actions) + [actions[-1]] * (action_horizon - len(actions))
        return actions

    @staticmethod
    def _stack_actions(actions) -> np.ndarray:
        return np.stack([np.asarray(action, dtype=np.float32) for action in actions], axis=0)
