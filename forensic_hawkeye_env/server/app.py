# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Forensic Hawkeye Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ForensicHawkeyeAction, ForensicHawkeyeObservation
    from .forensic_hawkeye_env_environment import ForensicHawkeyeEnvironment
except (ImportError, ModuleNotFoundError):
    from models import ForensicHawkeyeAction, ForensicHawkeyeObservation
    from server.forensic_hawkeye_env_environment import ForensicHawkeyeEnvironment


# Create the app with web interface and README integration
app = create_app(
    ForensicHawkeyeEnvironment,
    ForensicHawkeyeAction,
    ForensicHawkeyeObservation,
    env_name="forensic_hawkeye_env",
    max_concurrent_envs=3,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
