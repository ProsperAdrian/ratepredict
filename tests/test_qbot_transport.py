from __future__ import annotations

import unittest
from unittest.mock import patch

from app.config import Settings
from app.services.market_data import LiveQuoteService


class QbotTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            qbot_service_token="service-token-value",
            qbot_cf_access_client_id="client-id-value.access",
            qbot_cf_access_client_secret="client-secret-value",
        )
        self.service = LiveQuoteService(self.settings)

    def test_cloudflare_error_reports_ray_without_secret_material(self) -> None:
        self.service._http_get = lambda *args, **kwargs: (
            403,
            "<!DOCTYPE html><html class=\"no-js ie6 oldie\">blocked</html>",
            "httpx",
            {
                "cf-ray": "1234567890abcdef-LOS",
                "content-type": "text/html; charset=UTF-8",
                "server": "cloudflare",
            },
        )

        headers = {
            "x-service-token": self.settings.qbot_service_token,
            "CF-Access-Client-Id": self.settings.qbot_cf_access_client_id,
            "CF-Access-Client-Secret": self.settings.qbot_cf_access_client_secret,
        }
        with self.assertRaisesRegex(RuntimeError, "cf_ray=1234567890abcdef-LOS") as raised:
            self.service._request_json(
                self.settings.qbot_usdtngn_rate_url,
                headers=headers,
                follow_redirects=False,
            )

        message = str(raised.exception)
        self.assertIn("cf_access_headers=present", message)
        self.assertIn("service_token=present", message)
        self.assertNotIn("service-token-value", message)
        self.assertNotIn("client-id-value", message)
        self.assertNotIn("client-secret-value", message)
        self.assertNotIn("_prefix=", message)

    def test_http_transport_returns_normalized_diagnostic_headers(self) -> None:
        class Response:
            status_code = 200
            text = '{"status":"success"}'
            headers = {
                "CF-Ray": "1234567890abcdef-LOS",
                "Content-Type": "application/json",
            }

        class Client:
            def __init__(self, **kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                pass

            def get(self, url, headers):
                return Response()

        with patch("app.services.market_data.httpx.Client", Client):
            status, body, transport, response_headers = self.service._http_get(
                self.settings.qbot_usdtngn_rate_url,
                headers={"Accept": "application/json"},
                timeout=5,
                follow_redirects=False,
            )

        self.assertEqual(status, 200)
        self.assertEqual(body, '{"status":"success"}')
        self.assertEqual(transport, "httpx")
        self.assertEqual(response_headers["cf-ray"], "1234567890abcdef-LOS")


if __name__ == "__main__":
    unittest.main()
