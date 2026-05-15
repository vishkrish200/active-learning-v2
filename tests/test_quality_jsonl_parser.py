import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.preprocessing.quality import load_modal_jsonl_imu


class QualityJsonlParserTests(unittest.TestCase):
    def test_load_modal_jsonl_recovers_single_unknown_three_vector_as_acc_when_gyro_is_present(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "imu.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"t_us": 0, "acc": [1.0, 2.0, 3.0], "gyro": [0.1, 0.2, 0.3]}),
                        json.dumps({"t_us": 33333, "mangled_acc_key": [4.0, 5.0, 6.0], "gyro": [0.4, 0.5, 0.6]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            samples, timestamps = load_modal_jsonl_imu(path)

            self.assertEqual(samples.shape, (2, 6))
            self.assertEqual(samples[1].tolist(), [4.0, 5.0, 6.0, 0.4, 0.5, 0.6])
            self.assertEqual(timestamps.tolist(), [0.0, 0.033333])


if __name__ == "__main__":
    unittest.main()
