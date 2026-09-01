#!/usr/bin/env python3
"""
Push an already-generated URDF string onto a running node's ``robot_description``
parameter, live, with no process restart.

Why this exists instead of ``ros2 param set``: robot_state_publisher's
``add_on_set_parameters_callback``/``onParameterEvent`` (see
https://github.com/ros/robot_state_publisher/blob/ros2/src/robot_state_publisher.cpp)
re-parses the URDF and republishes fixed transforms as soon as its
``robot_description`` parameter changes, so *updating an existing fixed joint's
origin* (e.g. after re-baking a hand-eye or tool-TCP calibration into a xacro
file) takes effect immediately, without restarting robot_state_publisher. But
``ros2 param set``/``ros2 param load`` round-trip the value through YAML
parsing (see ``rclpy.parameter.get_parameter_value``), which breaks on a raw
URDF string because ``:`` inside XML comments/attributes is read as YAML
mapping syntax. This calls the node's ``~/set_parameters`` service directly
with a typed ``PARAMETER_STRING`` value, sidestepping that YAML round-trip
entirely.

This only republishes the *value* of an existing fixed joint's transform, not
a new kinematic structure: nodes that cache their own robot model at startup
(MoveIt's ``move_group``) will not see the change until they are restarted.

Typical use, after baking a calibration into a xacro file (see
``tool_tcp_cli.py`` / ``depthai_mount_cli.py`` / ``realsense_mount_cli.py``):

    xacro /path/to/top_level.urdf.xacro some_arg:=value > /tmp/robot_description.urdf
    python3 -m hand_eye_calibration.push_robot_description_cli \\
        --node /robot_state_publisher --urdf-file /tmp/robot_description.urdf
"""

from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


def push_robot_description(node_name: str, urdf: str, timeout_s: float = 15.0) -> tuple[bool, str]:
    """Call ``<node_name>/set_parameters`` with a raw ``robot_description`` string.

    Returns ``(ok, message)``; never raises for an unreachable service or a
    rejected parameter, so callers (e.g. launch_server) get a clean result to
    report back to the GUI instead of a traceback.
    """
    rclpy.init(args=[])
    try:
        node = Node("push_robot_description_cli")
        try:
            client = node.create_client(SetParameters, f"{node_name}/set_parameters")
            if not client.wait_for_service(timeout_sec=timeout_s):
                return False, f"service {node_name}/set_parameters not available"

            request = SetParameters.Request()
            request.parameters = [
                Parameter(
                    name="robot_description",
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_STRING, string_value=urdf
                    ),
                )
            ]
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
            if future.result() is None:
                return False, f"{node_name}/set_parameters call timed out"
            (result,) = future.result().results
            if not result.successful:
                return False, result.reason or "rejected by node"
            return True, "robot_description updated live"
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node",
        default="/robot_state_publisher",
        help="Node to update (default: /robot_state_publisher)",
    )
    parser.add_argument(
        "--urdf-file",
        required=True,
        help="Path to a file containing the full, already-xacro-processed URDF XML",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)

    with open(args.urdf_file, "r", encoding="utf-8") as f:
        urdf = f.read()
    if not urdf.strip():
        print("empty URDF file", file=sys.stderr)
        return 1

    ok, message = push_robot_description(args.node, urdf, timeout_s=args.timeout)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
