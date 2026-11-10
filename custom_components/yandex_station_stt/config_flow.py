"""Config flow for Yandex Station STT integration."""

from __future__ import annotations

import logging
from typing import Any
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.data_entry_flow import FlowResult
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe

_LOGGER = logging.getLogger(__name__)

DOMAIN = "yandex_station_stt"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 10300


class YandexSttConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Yandex Station STT."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            host = user_input[CONF_HOST]
            port = user_input[CONF_PORT]

            try:
                # Test connection to Wyoming server
                client = AsyncTcpClient(host, port)
                await client.connect()
                await client.write_event(Describe().event())
                await client.disconnect()
            except Exception as err:
                _LOGGER.warning("Could not connect to Wyoming server at %s:%s: %s", host, port, err)
                errors["base"] = "cannot_connect"

            if not errors:
                await self.async_set_unique_id(f"{host}:{port}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"Yandex Station STT ({host}:{port})",
                    data=user_input,
                )

        schema = vol.Schema(
            {
                vol.Required(CONF_HOST, default=DEFAULT_HOST): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )
