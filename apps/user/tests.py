from django.test import TestCase

from rest_framework.test import APIClient


# Create your tests here.
class UserTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_user(self):
        response = self.client.get("/user/")
        self.assertEqual(response.status_code, 200)
