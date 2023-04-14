import pytest
from torch.utils.tensorboard import SummaryWriter
from vesuvius.trackers import BaseTracker, TrackerAvg


def test_TrackerAvg_update():
    tracker = TrackerAvg(tag="test_loss", summary_writer=SummaryWriter())

    losses = [1.0, 2.0, 3.0, 4.0, 5.0]
    batch_size = 10

    total_loss = 0
    total_items = 0

    for loss in losses:
        tracker.update(loss, batch_size)
        total_loss += loss * batch_size
        total_items += batch_size
        assert tracker.value == total_loss
        assert tracker.i == total_items
        assert tracker.average == total_loss / total_items


def test_TrackerAvg_log(mocker):
    tracker = TrackerAvg(tag="test_loss", summary_writer=SummaryWriter())
    tracker.value = 100.0
    tracker.i = 10

    mocker.spy(tracker.summary_writer, "add_scalar")

    tracker.log(1000)

    tracker.summary_writer.add_scalar.assert_called_once_with("test_loss", 10.0, 1000)


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_tracker.py"])
