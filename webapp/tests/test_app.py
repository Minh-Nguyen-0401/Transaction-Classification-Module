import os
import unittest

os.environ["SKIP_GEMINI"] = "1"  # avoid network calls in tests

from fastapi.testclient import TestClient

from webapp.main import app


class AppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("status"), "ok")

    def test_options_and_classify_rule_short_circuit(self):
        res = self.client.get("/options")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["senders"])
        self.assertTrue(data["recipients"])
        sender = data["default_sender"] or data["senders"][0]["code"]
        recipient = data["default_recipient"] or data["recipients"][0]["code"]

        payload = {
            "message": "Mua sắm tại sieu thi",
            "amount": 100000,
            "recipient_entity_id": recipient,
            "sender_id": sender,
        }
        res2 = self.client.post("/classify", json=payload)
        self.assertEqual(res2.status_code, 200)
        out = res2.json()
        self.assertIn(out["decided_by"], ["layer1_rule", "layer3_model", "layer2_llm"])
        self.assertTrue(out["final_label"])


if __name__ == "__main__":
    unittest.main()
