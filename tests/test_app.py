import unittest
from app import app

class FlaskTestCase(unittest.TestCase):

    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 404)

    def test_predict(self):
        tester = app.test_client(self)
        with open("tests/sample_dog.jpg", "rb") as img:
            response = tester.post('/predict', data={"file": img})
            self.assertEqual(response.status_code, 200)
            self.assertIn("breed", response.json)

if __name__ == '__main__':
    unittest.main()
