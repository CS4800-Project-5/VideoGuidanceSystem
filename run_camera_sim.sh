#!/usr/bin/env bash

set -Eeuo pipefail

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
ROS_SETUP="/opt/ros/humble/setup.bash"
CAMERA_TOPIC="/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
LOG_DIR="${LOG_DIR:-$PWD/sim-logs}"

if [[ ! -d "$PX4_DIR" ]]; then
    echo "Error: PX4 directory not found: $PX4_DIR" >&2
    echo "Set PX4_DIR to your PX4-Autopilot directory if needed." >&2
    exit 1
fi

if [[ ! -f "$ROS_SETUP" ]]; then
    echo "Error: ROS 2 setup file not found: $ROS_SETUP" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# Load ROS 2 Humble into this shell and avoid forcing a middleware implementation.
source "$ROS_SETUP"
unset RMW_IMPLEMENTATION

PX4_PID=""
BRIDGE_PID=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    echo
    echo "Stopping simulation processes..."

    if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -- "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi

    # PX4/Gazebo may create child processes, so terminate the whole session.
    if [[ -n "$PX4_PID" ]] && kill -0 "$PX4_PID" 2>/dev/null; then
        kill -- "-$PX4_PID" 2>/dev/null || kill "$PX4_PID" 2>/dev/null || true
        wait "$PX4_PID" 2>/dev/null || true
    fi

    exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting PX4 SITL and Gazebo..."
echo "PX4 logs: $LOG_DIR/px4-gazebo.log"

(
    cd "$PX4_DIR"
    exec env \
        LIBGL_ALWAYS_SOFTWARE=1 \
        GALLIUM_DRIVER=llvmpipe \
        MESA_GL_VERSION_OVERRIDE=3.3 \
        make px4_sitl gz_x500_mono_cam
) >"$LOG_DIR/px4-gazebo.log" 2>&1 &
PX4_PID=$!

echo "Waiting for the Gazebo camera topic..."
camera_ready=false
for ((attempt = 1; attempt <= 60; attempt++)); do
    # This is a Gazebo Transport topic before the image bridge starts.
    if command -v gz >/dev/null 2>&1 && \
        gz topic -l 2>/dev/null | grep -Fxq "$CAMERA_TOPIC"; then
        camera_ready=true
        break
    fi

    if ! kill -0 "$PX4_PID" 2>/dev/null; then
        echo "Error: PX4/Gazebo stopped unexpectedly." >&2
        echo "Check $LOG_DIR/px4-gazebo.log for details." >&2
        exit 1
    fi

    sleep 1
done

if [[ "$camera_ready" != true ]]; then
    echo "Error: Camera topic did not appear within 60 seconds." >&2
    echo "Check $LOG_DIR/px4-gazebo.log for details." >&2
    exit 1
fi

echo "Starting ROS-Gazebo image bridge..."
ros2 run ros_gz_image image_bridge \
    "$CAMERA_TOPIC" \
    --ros-args -p qos:=sensor_data \
    >"$LOG_DIR/image-bridge.log" 2>&1 &
BRIDGE_PID=$!

sleep 2

if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "Error: Image bridge stopped unexpectedly." >&2
    echo "Check $LOG_DIR/image-bridge.log for details." >&2
    exit 1
fi

echo "Opening the camera viewer. Press Ctrl+C to stop everything."
ros2 run image_tools showimage \
    --ros-args \
    -r "image:=$CAMERA_TOPIC"

