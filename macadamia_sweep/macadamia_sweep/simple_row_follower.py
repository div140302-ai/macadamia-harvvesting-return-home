#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String


class SimpleRowFollower(Node):
    def __init__(self):
        super().__init__("simple_row_follower")

        # Parameter: starting side the row is on. "right" or "left".
        # Override at launch: --ros-args -p start_side:=left
        self.declare_parameter("start_side", "right")
        self.start_side = str(self.get_parameter("start_side").value).lower()
        if self.start_side not in ("right", "left"):
            self.get_logger().warn(
                f"Unknown start_side '{self.start_side}', defaulting to 'right'"
            )
            self.start_side = "right"

        # Parameter: rotation (radians) to add to raw lidar angles so that
        # the code's "0 deg = forward" convention matches the physical
        # orientation of the lidar on this robot. If the lidar is mounted
        # rotated 180 deg (an object physically in front of the robot
        # reports at +/-180 deg in raw scan angles), use math.pi here.
        # Verify with the lidar_check diagnostic.
        self.declare_parameter("lidar_yaw_offset_deg", 180.0)
        self.lidar_yaw_offset = math.radians(
            float(self.get_parameter("lidar_yaw_offset_deg").value)
        )

        # ---------------------------------------------------------------
        # Multi-row sweep.
        #
        # Mission: sweep one side of row 1, U-turn at the far end, sweep
        # its other side back. During that return pass the NEXT row sits on
        # the opposite side of the robot; if it is seen consistently, the
        # robot clears the row start, turns 180 deg in place toward the
        # next row, and repeats the identical sweep on it. Every row is
        # swept exactly twice (once per side). Nothing about row spacing
        # or row count is hardcoded.
        #
        # max_rows > 0 -> FIXED COUNT (selected behaviour): sweep exactly
        #                 that many rows, turning to the next row after each
        #                 one REGARDLESS of what perception sees. Set with
        #                 --ros-args -p max_rows:=4.
        # max_rows = 0 -> AUTO: continue only while a next row is detected on
        #                 the opposite side during the return pass.
        # ---------------------------------------------------------------
        self.declare_parameter("max_rows", 0)
        self.max_rows = int(self.get_parameter("max_rows").value)
        # How many control ticks (10 Hz) the opposite-side row fit must
        # succeed during FOLLOW_BACK before we commit to a next row.
        self.declare_parameter("next_row_min_hits", 8)
        self.next_row_min_hits = int(self.get_parameter("next_row_min_hits").value)
        # A genuine NEXT row sits (row spacing - follow offset) away on the
        # opposite side during the return pass (~0.30 m for 0.70 m rows).
        # Anything farther than this is NOT the next row (e.g. an already
        # swept row seen across a gap) - prevents re-sweeping forever.
        self.declare_parameter("next_row_max_dist", 0.60)
        self.next_row_max_dist = float(
            self.get_parameter("next_row_max_dist").value
        )
        self.next_row_hits = 0
        self.rows_completed = 0

        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel_nav", 10)
        self.status_pub = self.create_publisher(String, "/snc_status", 10)

        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 20)
        self.create_subscription(Empty, "/sweep_start", self.start_callback, 10)
        self.create_subscription(Empty, "/sweep_stop", self.stop_callback, 10)
        self.create_subscription(Empty, "/return_home", self.return_home_callback, 10)

        self.latest_scan: Optional[LaserScan] = None
        self.odom_x: Optional[float] = None
        self.odom_y: Optional[float] = None
        self.odom_yaw: Optional[float] = None

        # Return-home bookkeeping. These are saved at /sweep_start and used
        # after the row/nut mission so the robot drives back to where it began.
        self.home_x: Optional[float] = None
        self.home_y: Optional[float] = None
        self.home_yaw: Optional[float] = None

        self.started = False
        self.state = "WAITING"
        self.state_start_time = self.get_clock().now()
        self.last_row_seen_time = self.get_clock().now()
        self.seen_row_this_pass = False

        # Side the row is on for the CURRENT pass.
        # The arc U-turn LOOPS AROUND the last tree, crossing the row line.
        # After it, the row reappears on the SAME side of the robot - not
        # flipped.
        self.current_side = self.start_side

        # CLEAR_END / arc bookkeeping.
        self.clear_start_x: Optional[float] = None
        self.clear_start_y: Optional[float] = None
        self.arc_start_yaw: Optional[float] = None
        self.arc_last_yaw: Optional[float] = None
        self.arc_accumulated_yaw = 0.0

        # ---------------------------------------------------------------
        # Heading anchor.
        #
        # The robot is placed parallel to the row at /sweep_start. So the
        # odom yaw AT THAT INSTANT is the row's direction in the odom frame.
        # No EMA, no "only update when visible" rule, no drift accumulation
        # from steering wobble - it's a single snapshot.
        #
        # outbound_yaw: row direction when going OUT. Set once at sweep_start.
        # return_target_yaw: outbound_yaw + pi, the heading we want after
        #                    the arc. Computed deterministically.
        # ---------------------------------------------------------------
        self.outbound_yaw: Optional[float] = None
        self.return_target_yaw: Optional[float] = None

        # ALIGN gains
        self.k_align = 1.2
        self.min_align_angular = 0.08

        # -----------------------------
        # Row-following tuning
        # -----------------------------
        self.desired_side_distance = 0.40
        self.too_close_side = 0.25
        # 1.20 m (was 0.80). At a 0.40 m follow offset with 0.70 m tree
        # spacing, 0.80 m only catches a second tree marginally (worst case
        # 0.81 m), so the fit flickered - and the next-row check on the
        # return pass starved. 1.20 m sees 2-3 trees continuously. Cross-row
        # contamination is prevented by select_nearest_row, and
        # next_row_max_dist stops a far row from counting as "next".
        self.declare_parameter("tree_max_range", 1.20)
        self.tree_max_range = float(self.get_parameter("tree_max_range").value)
        self.lidar_min_range = 0.25

        self.emergency_stop_distance = 0.18

        self.forward_speed = 0.06
        self.search_speed = 0.03

        self.k_xtrack = 1.2
        self.k_heading = 1.4
        self.max_follow_angular = 0.30

        self.start_straight_duration = 1.5

        # -----------------------------
        # Row detection (cluster-then-fit)
        # -----------------------------
        self.cluster_gap = 0.15
        self.tree_max_extent = 0.15
        # With several rows in lidar range the robot can see MULTIPLE
        # candidate rows at once. Tree clusters are therefore grouped by
        # their offset perpendicular to the known row direction, and only
        # the NEAREST group on the requested side feeds the line fit - the
        # next row over can never contaminate it. The gap just has to be
        # smaller than the real row spacing (0.35 m suits 0.70 m rows and
        # anything wider).
        self.declare_parameter("row_group_gap", 0.35)
        self.row_group_gap = float(self.get_parameter("row_group_gap").value)
        # Ignore returns within this lateral band of the robot's row axis
        # (they can't be unambiguously assigned to a side).
        self.min_side_offset = 0.05
        self.min_trees_for_fit = 2
        self.front_width = math.radians(35)
        self.max_tree_points = 80

        self.min_pass_duration = 10.0
        self.max_pass_duration = 90.0
        self.row_lost_timeout = 2.0

        # CLEAR_END (0.15 puts the arc centre right at the row end with
        # ~0.25 m centre clearance to the last tree - see arc note below).
        self.declare_parameter("clear_end_distance", 0.15)
        self.clear_end_distance = float(
            self.get_parameter("clear_end_distance").value
        )
        self.clear_end_max_time = 8.0

        # Arc U-turn - TIGHT.
        # Geometry (following at 0.40 m from the row line):
        #   end lateral offset = 0.40 - 2*r from the row line.
        #   r = 0.55 (old) -> ends 0.70 m on the other side = ON the next
        #     row line for 0.70 m spacing. Too wide.
        #   r = 0.40 -> ends 0.40 m on the other side, max forward
        #     excursion clear_end + r = 0.55 m past the last tree, and the
        #     whole arc stays within +/-0.40 m of the row line. With
        #     clear_end = 0.15 the arc centre sits ~0.15 m from the last
        #     tree, so the robot orbits it with r - 0.15 = 0.25 m centre
        #     clearance (robot half-width 0.125 + noodle 0.06 = 0.19 m).
        self.declare_parameter("arc_radius", 0.40)
        self.arc_radius = float(self.get_parameter("arc_radius").value)
        self.arc_linear_speed = 0.06
        self.arc_max_yaw = math.radians(220.0)
        self.arc_distance_gain = 0.6

        nominal_omega = self.arc_linear_speed / self.arc_radius
        self.arc_max_duration = self.arc_max_yaw / nominal_omega + 5.0

        # ALIGN
        self.align_max_angular = 0.30
        self.align_parallel_tol = math.radians(12.0)
        self.align_max_duration = 8.0

        # TURN_NEXT (in-place 180 deg toward the next row). A full 180 at
        # align_max_angular takes ~10.5 s, so this needs its own, longer
        # cap than ALIGN (which only trims the small residual after the arc).
        self.turn_next_max_duration = 18.0

        self.recovery_angular = 0.20

        # -----------------------------
        # RETURN_HOME
        # -----------------------------
        # After the row sweep / nut collection run is finished, the robot
        # returns to the odometry position recorded at /sweep_start.
        self.declare_parameter("return_home_enabled", True)
        self.return_home_enabled = bool(
            self.get_parameter("return_home_enabled").value
        )
        self.declare_parameter("return_goal_tolerance", 0.12)
        self.return_goal_tolerance = float(
            self.get_parameter("return_goal_tolerance").value
        )
        self.declare_parameter("return_yaw_tolerance_deg", 12.0)
        self.return_yaw_tolerance = math.radians(
            float(self.get_parameter("return_yaw_tolerance_deg").value)
        )
        self.declare_parameter("return_max_duration", 120.0)
        self.return_max_duration = float(
            self.get_parameter("return_max_duration").value
        )
        self.return_linear_speed = 0.07
        self.return_max_angular = 0.35
        self.return_heading_slowdown = math.radians(35.0)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.publish_status("Simple row follower ready. Publish /sweep_start to begin.")
        self.get_logger().info(
            f"Ready (multi-row). CLEAR={self.clear_end_distance:.2f}m, "
            f"ARC r={self.arc_radius:.2f}m v={self.arc_linear_speed:.2f}m/s, "
            f"start_side={self.start_side}, "
            f"max_rows={self.max_rows or 'unlimited'}, "
            f"row_group_gap={self.row_group_gap:.2f}m, "
            f"lidar_yaw_offset={math.degrees(self.lidar_yaw_offset):+.0f}deg, "
            f"return_home={self.return_home_enabled}, manual_topic=/return_home"
        )

    # -----------------------------
    # Callbacks
    # -----------------------------

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny, cosy)

    def start_callback(self, _msg: Empty):
        # GUARD: `ros2 topic pub /sweep_start ...` WITHOUT `--once` keeps
        # publishing at 1 Hz; each repeat would re-snapshot the heading
        # anchor at the robot's CURRENT yaw (garbage mid-turn) and reset
        # the state machine. Over a multi-row mission that is fatal, so
        # ignore restarts while running.
        if self.started:
            self.get_logger().warn(
                "sweep_start ignored - sweep already running. "
                "Publish /sweep_stop first to restart."
            )
            return
        self.started = True
        self.rows_completed = 0
        self.next_row_hits = 0
        self.state = "FOLLOW_OUT"
        self.state_start_time = self.get_clock().now()
        self.last_row_seen_time = self.get_clock().now()
        self.seen_row_this_pass = False
        self.current_side = self.start_side
        self.clear_start_x = None
        self.clear_start_y = None
        self.arc_start_yaw = None
        self.arc_last_yaw = None
        self.arc_accumulated_yaw = 0.0

        # Save the starting pose for RETURN_HOME. If odom is not ready yet,
        # control_loop will lazily fill this before the robot has moved far.
        if self.odom_x is not None and self.odom_y is not None:
            self.home_x = self.odom_x
            self.home_y = self.odom_y
            self.home_yaw = self.odom_yaw
            self.get_logger().info(
                f"Home pose saved: x={self.home_x:+.2f}, y={self.home_y:+.2f}, "
                f"yaw={math.degrees(self.home_yaw or 0.0):+.1f}deg"
            )
        else:
            self.home_x = None
            self.home_y = None
            self.home_yaw = None
            self.get_logger().warn(
                "Odom not ready at sweep_start - home pose will be saved on first odom tick."
            )

        # THE key change: snapshot the row direction NOW. The robot is
        # parallel to the row at this instant, so odom_yaw is the row's
        # direction in odom. Return heading is exactly the opposite.
        if self.odom_yaw is not None:
            self.outbound_yaw = self.odom_yaw
            self.return_target_yaw = self.normalize_angle(self.odom_yaw + math.pi)
            self.get_logger().info(
                f"Anchored row heading: outbound={math.degrees(self.outbound_yaw):+.1f}deg "
                f"return_target={math.degrees(self.return_target_yaw):+.1f}deg"
            )
        else:
            # Odom not ready yet - will be set lazily by control_loop on
            # the first tick odom arrives.
            self.outbound_yaw = None
            self.return_target_yaw = None
            self.get_logger().warn(
                "Odom not ready at sweep_start - heading anchor will be set on first odom tick."
            )

        self.publish_status(f"Started: line-fit following on {self.current_side.upper()}")
        self.get_logger().info(f"Sweep start received (side={self.current_side})")

    def stop_callback(self, _msg: Empty):
        self.started = False
        self.state = "STOPPED"
        self.stop_robot()
        self.publish_status("Manual stop received")
        self.get_logger().warn("Sweep stop received")

    def return_home_callback(self, _msg: Empty):
        """Manual trigger: publish /return_home to abandon the current sweep and drive home."""
        # If home was not saved at /sweep_start because odometry was late,
        # try to save it now only if the robot has not started moving yet.
        # Normally this will already be set by start_callback/control_loop.
        if self.home_x is None or self.home_y is None:
            self.get_logger().warn(
                "return_home requested, but no home pose has been saved. "
                "Publish /sweep_start first so the robot knows where home is."
            )
            self.publish_status("Cannot RETURN_HOME: no saved home pose. Publish /sweep_start first.")
            return

        self.started = True
        self.next_row_hits = 0
        self.clear_start_x = None
        self.clear_start_y = None
        self.arc_start_yaw = None
        self.arc_last_yaw = None
        self.arc_accumulated_yaw = 0.0
        self.stop_robot()
        self.set_state(
            "RETURN_HOME",
            f"Manual return-home requested. Returning to x={self.home_x:+.2f}, y={self.home_y:+.2f}."
        )
        self.get_logger().warn("Manual return-home requested")

    # -----------------------------
    # Helpers
    # -----------------------------

    def publish_status(self, text: str):
        m = String()
        m.data = text
        self.status_pub.publish(m)

    def elapsed_in_state(self) -> float:
        return (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9

    def time_since_row_seen(self) -> float:
        return (self.get_clock().now() - self.last_row_seen_time).nanoseconds / 1e9

    def set_state(self, new_state: str, status: str = ""):
        self.state = new_state
        self.state_start_time = self.get_clock().now()
        if status:
            self.publish_status(status)
        self.get_logger().info(f"State changed to {new_state}")

    def publish_cmd(self, linear: float, angular: float):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.twist.linear.x = float(linear)
        m.twist.angular.z = float(angular)
        self.cmd_pub.publish(m)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def normalize_angle(self, a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def valid_range(self, r: float, scan: LaserScan) -> bool:
        return (
            not math.isnan(r)
            and not math.isinf(r)
            and scan.range_min <= r <= scan.range_max
            and r >= self.lidar_min_range
        )

    def collect_row_side_points(self, side: str) -> List[Tuple[float, float, float]]:
        scan = self.latest_scan
        if scan is None:
            return []
        if side == "right":
            min_a = math.radians(-170.0)
            max_a = math.radians(20.0)
        else:
            min_a = math.radians(-20.0)
            max_a = math.radians(170.0)

        pts: List[Tuple[float, float, float]] = []
        for i, r in enumerate(scan.ranges):
            # Apply lidar mounting offset: raw scan angle PLUS the offset
            # gives the angle in the robot's base_link frame (forward = 0).
            angle = scan.angle_min + i * scan.angle_increment + self.lidar_yaw_offset
            angle = self.normalize_angle(angle)
            if not (min_a <= angle <= max_a):
                continue
            if not self.valid_range(r, scan):
                continue
            if r > self.tree_max_range:
                continue
            pts.append((r * math.cos(angle), r * math.sin(angle), angle))
        pts.sort(key=lambda p: p[2])
        return pts

    def cluster_to_trees(
        self, pts: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, int]]:
        if not pts:
            return []
        clusters: List[List[Tuple[float, float, float]]] = []
        current: List[Tuple[float, float, float]] = [pts[0]]
        for prev, p in zip(pts, pts[1:]):
            d = math.hypot(p[0] - prev[0], p[1] - prev[1])
            if d <= self.cluster_gap:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)

        trees: List[Tuple[float, float, int]] = []
        for c in clusters:
            xs = [q[0] for q in c]
            ys = [q[1] for q in c]
            extent = max(max(xs) - min(xs), max(ys) - min(ys))
            if extent > self.tree_max_extent:
                continue
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            trees.append((mx, my, len(c)))
        return trees

    @staticmethod
    def opposite(side: str) -> str:
        return "left" if side == "right" else "right"

    def row_dir_robot_frame(self) -> Optional[float]:
        """Known row direction expressed in the robot frame.

        The heading anchor gives the row direction in odom; subtracting the
        current yaw moves it into base_link. A row is a LINE, so the value
        only matters mod pi - that is fine for projecting trees onto the
        across-row axis, and it stays valid in any robot orientation,
        including halfway around the U-turn.
        """
        if self.outbound_yaw is None or self.odom_yaw is None:
            return None
        return self.normalize_angle(self.outbound_yaw - self.odom_yaw)

    def select_nearest_row(
        self, trees: List[Tuple[float, float, int]], side: str
    ) -> List[Tuple[float, float, int]]:
        """Keep only the NEAREST row of trees on the requested side.

        With several rows in range the raw cluster list may contain trees
        from two (or more) parallel rows. Each tree is projected onto the
        axis perpendicular to the known row direction; trees are grouped
        along that axis (1-D clustering with row_group_gap) and the group
        with the smallest mean |offset| on the correct side wins. Trees of
        the next row sit a full row-spacing away on that axis, so they end
        up in a different group and never contaminate the fit.
        """
        if not trees:
            return []
        a = self.row_dir_robot_frame()
        if a is None:
            a = 0.0  # robot parallel to row (true at start, before anchor)
        # A row is a LINE - its direction is only defined mod pi. Use the
        # representative closest to the robot's forward axis so that
        # perp > 0 always means the ROBOT's left ('side' is a robot-frame
        # concept). Without this, the sign flips on the return pass and
        # every tree gets assigned to the wrong side.
        if a > math.pi / 2.0:
            a -= math.pi
        elif a < -math.pi / 2.0:
            a += math.pi
        sin_a = math.sin(a)
        cos_a = math.cos(a)

        tagged: List[Tuple[float, float, float, int]] = []
        for (tx, ty, n) in trees:
            # perp > 0 -> left of the row axis through the robot.
            perp = -tx * sin_a + ty * cos_a
            if side == "right" and perp > -self.min_side_offset:
                continue
            if side == "left" and perp < self.min_side_offset:
                continue
            tagged.append((perp, tx, ty, n))
        if not tagged:
            return []

        tagged.sort(key=lambda t: t[0])
        groups: List[List[Tuple[float, float, float, int]]] = [[tagged[0]]]
        for prev, cur in zip(tagged, tagged[1:]):
            if abs(cur[0] - prev[0]) <= self.row_group_gap:
                groups[-1].append(cur)
            else:
                groups.append([cur])

        best = min(
            groups,
            key=lambda g: abs(sum(t[0] for t in g) / len(g)),
        )
        return [(tx, ty, n) for (_, tx, ty, n) in best]

    def fit_row_line(self, side: str) -> Optional[Tuple[float, float, int]]:
        pts = self.collect_row_side_points(side)
        trees = self.cluster_to_trees(pts)
        trees = self.select_nearest_row(trees, side)
        if len(trees) < self.min_trees_for_fit:
            return None
        cx = [t[0] for t in trees]
        cy = [t[1] for t in trees]
        n = len(trees)
        sx = sum(cx)
        sy = sum(cy)
        sxx = sum(x * x for x in cx)
        sxy = sum(x * y for x, y in zip(cx, cy))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-6:
            mean_y = sy / n
            return (math.pi / 2.0, mean_y, n)
        m = (n * sxy - sx * sy) / denom
        c = (sy - m * sx) / n
        line_angle = math.atan(m)
        perp = c / math.sqrt(1.0 + m * m)
        return (line_angle, perp, n)

    def side_cone_points(self, side: str) -> List[Tuple[float, float]]:
        pts = self.collect_row_side_points(side)
        trees = self.cluster_to_trees(pts)
        # Same nearest-row gating as the fit, so "last tree abeam" is
        # judged against the CURRENT row only, not a neighbouring one.
        trees = self.select_nearest_row(trees, side)
        return [(t[0], t[1]) for t in trees]

    def get_front_distance(self) -> float:
        scan = self.latest_scan
        if scan is None:
            return 10.0
        half_w = self.front_width / 2.0
        vals = []
        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment + self.lidar_yaw_offset
            if abs(self.normalize_angle(angle - 0.0)) > half_w:
                continue
            if not self.valid_range(r, scan):
                continue
            if r > 3.0:
                continue
            vals.append(r)
        return min(vals) if vals else 10.0

    def row_visible(self, side: str) -> Tuple[bool, Optional[Tuple[float, float, int]]]:
        fit = self.fit_row_line(side)
        return (fit is not None), fit

    def last_tree_abeam_or_behind(self, side: str) -> bool:
        if not self.seen_row_this_pass:
            return False
        pts = self.side_cone_points(side)
        if not pts:
            return False
        ahead_threshold = 0.05
        max_x = max(p[0] for p in pts)
        return max_x <= ahead_threshold

    def mark_row_seen_if_visible(self, visible: bool):
        if visible:
            self.last_row_seen_time = self.get_clock().now()
            self.seen_row_this_pass = True

    # -----------------------------
    # Line-fit follow
    # -----------------------------

    def follow_side(self, side: str):
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP: front={front:.2f} m")
            return

        visible, fit = self.row_visible(side)
        self.mark_row_seen_if_visible(visible)

        if self.elapsed_in_state() < self.start_straight_duration:
            self.publish_cmd(self.forward_speed, 0.0)
            self.publish_status(
                f"START STRAIGHT | row_visible={visible} | front={front:.2f}m"
            )
            return

        if not visible:
            self.publish_cmd(self.search_speed, 0.0)
            self.publish_status(
                f"ROW NOT CONFIRMED {side.upper()} | creeping forward | "
                f"front={front:.2f}m"
            )
            return

        line_angle, perp, n = fit  # type: ignore[assignment]

        desired_perp = (-self.desired_side_distance
                        if side == "right" else self.desired_side_distance)
        move_left = perp - desired_perp
        heading_error = line_angle

        angular = self.k_xtrack * move_left + self.k_heading * heading_error
        angular = max(-self.max_follow_angular,
                      min(self.max_follow_angular, angular))
        linear = self.forward_speed

        actual_dist = abs(perp)
        if actual_dist < self.too_close_side:
            linear = self.search_speed
            status = f"TOO CLOSE {side.upper()} d={actual_dist:.2f}m"
        else:
            status = f"FOLLOWING {side.upper()} d={actual_dist:.2f}m"

        self.publish_cmd(linear, angular)
        self.publish_status(
            f"{status} | "
            f"perp={perp:+.2f} target={desired_perp:+.2f} | "
            f"line_ang={math.degrees(line_angle):+.0f}deg | "
            f"n={n} | front={front:.2f}m | ang={angular:+.2f}"
        )

    def should_end_pass(self) -> bool:
        elapsed = self.elapsed_in_state()
        if elapsed < self.min_pass_duration:
            return False
        if self.seen_row_this_pass and self.time_since_row_seen() >= self.row_lost_timeout:
            return True
        if elapsed >= self.max_pass_duration:
            return True
        return False

    # -----------------------------
    # CLEAR_END
    # -----------------------------

    def clear_straight(self, next_state: str, label: str):
        """Drive straight until clear_end_distance past the row end, then
        hand over to next_state. Used both at the FAR end (-> ARC_TURN)
        and at the START end before a row transition (-> TURN_NEXT)."""
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during {label}: front={front:.2f}m")
            return

        if self.clear_start_x is None and self.odom_x is not None:
            self.clear_start_x = self.odom_x
            self.clear_start_y = self.odom_y

        if self.clear_start_x is None or self.odom_x is None:
            self.publish_cmd(self.forward_speed, 0.0)
            self.publish_status(f"{label} (no odom yet)")
            return

        dx = self.odom_x - self.clear_start_x
        dy = self.odom_y - self.clear_start_y
        traveled = math.hypot(dx, dy)

        if traveled >= self.clear_end_distance:
            self.set_state(
                next_state,
                f"Cleared {traveled:.2f}m past row end. -> {next_state}."
            )
            return

        if self.elapsed_in_state() > self.clear_end_max_time:
            self.set_state(
                next_state,
                f"{label} time cap, -> {next_state} anyway."
            )
            return

        self.publish_cmd(self.forward_speed, 0.0)
        self.publish_status(
            f"{label} {traveled:.2f}/{self.clear_end_distance:.2f}m | front={front:.2f}m"
        )

    # -----------------------------
    # Arc U-turn
    # -----------------------------

    def arc_turn(self):
        sign = -1.0 if self.current_side == "right" else +1.0

        if self.odom_yaw is not None:
            if self.arc_last_yaw is None:
                self.arc_start_yaw = self.odom_yaw
                self.arc_last_yaw = self.odom_yaw
            else:
                # SIGNED accumulation in the commanded turn direction:
                # abs() would count yaw NOISE as progress (a "180 deg" arc
                # can exit after ~120 deg of true rotation with just 1 deg
                # of per-reading jitter). Signed deltas let jitter cancel.
                d = self.normalize_angle(self.odom_yaw - self.arc_last_yaw)
                self.arc_accumulated_yaw += sign * d
                self.arc_last_yaw = self.odom_yaw

        yaw_deg = math.degrees(self.arc_accumulated_yaw)

        v = self.arc_linear_speed
        base_omega = v / self.arc_radius

        # PURE ODOMETRY ARC - no lidar correction.
        # The old lidar distance-correction measured "distance to the row"
        # mid-arc, but halfway around the turn the robot straddles the row
        # line: its own row's trees are excluded (|perp| ~ 0) and the
        # NEIGHBOURING row becomes the nearest fit, so the correction
        # tightened the turn into the last tree (offline sim showed a
        # near-collision, 3 cm clearance). A constant-omega arc has clean,
        # provable geometry: with clear_end=0.15 and r=0.40 the robot
        # orbits the last tree with ~0.25 m centre clearance.
        omega = base_omega

        # PRIMARY exit: rotated 180 deg from arc start.
        arc_target_yaw = math.pi
        if self.arc_accumulated_yaw >= arc_target_yaw:
            self.set_state(
                "ALIGN",
                f"Arc 180 deg complete at +{yaw_deg:.0f}. Aligning to row."
            )
            return

        # Safety cap.
        if (self.arc_accumulated_yaw >= self.arc_max_yaw
                or self.elapsed_in_state() >= self.arc_max_duration):
            self.set_state(
                "ALIGN",
                f"Arc capped at {yaw_deg:.0f} deg. Aligning to row."
            )
            return

        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.publish_cmd(0.0, sign * base_omega)
            self.publish_status(
                f"ARC (front block {front:.2f}m) | yaw +{yaw_deg:.0f}deg"
            )
            return

        self.publish_cmd(v, sign * omega)
        self.publish_status(
            f"ARC | yaw +{yaw_deg:.0f}/180deg | r={self.arc_radius:.2f}m | "
            f"front={front:.2f}m"
        )

    # -----------------------------
    # ALIGN - heading-anchored. NO line-fit dependency.
    # -----------------------------

    def align_to_row(self):
        """Rotate in place until odom_yaw matches return_target_yaw.

        We ONLY use the heading anchor recorded at /sweep_start. No line fit,
        no perception ambiguity. Geometric guarantee:
          - At sweep_start, robot was parallel to row, facing along OUT.
          - return_target_yaw = sweep_start_yaw + pi exactly.
          - When odom_yaw matches return_target_yaw, robot is parallel to
            the row, facing along BACK. Period.

        Previously this state mixed in the line-fit angle for "fine tuning",
        but line_angle is in (-90, +90) and can't disambiguate the two
        parallel orientations - that's what made the robot settle facing
        the wrong way. Heading-only termination is unambiguous.
        """
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during ALIGN: front={front:.2f}m")
            return

        if self.return_target_yaw is None or self.odom_yaw is None:
            # Should not happen because we snapshot at sweep_start, but if
            # it does, abort gracefully.
            self.publish_cmd(0.0, 0.0)
            self.publish_status(
                "ALIGN cannot proceed: missing heading anchor or odom."
            )
            if self.elapsed_in_state() > self.align_max_duration:
                self._enter_follow_back("ALIGN aborted - no heading reference.")
            return

        # Pure heading control. err > 0 -> target is "more CCW" -> turn left.
        err = self.normalize_angle(self.return_target_yaw - self.odom_yaw)

        if abs(err) < self.align_parallel_tol:
            self._enter_follow_back(
                f"ALIGN done. odom_yaw={math.degrees(self.odom_yaw):+.1f} "
                f"target={math.degrees(self.return_target_yaw):+.1f} "
                f"err={math.degrees(err):+.1f}deg. Sweeping back."
            )
            return

        mag = min(self.align_max_angular,
                  max(self.min_align_angular, self.k_align * abs(err)))
        angular = math.copysign(mag, err)
        self.publish_cmd(0.0, angular)
        self.publish_status(
            f"ALIGN | odom={math.degrees(self.odom_yaw):+.0f}deg "
            f"target={math.degrees(self.return_target_yaw):+.0f}deg "
            f"err={math.degrees(err):+.0f}deg (tol "
            f"{math.degrees(self.align_parallel_tol):.0f}) | rot={angular:+.2f}"
        )

        if self.elapsed_in_state() > self.align_max_duration:
            self._enter_follow_back(
                f"ALIGN timeout. err={math.degrees(err):+.0f}deg."
            )

    def _enter_follow_back(self, status: str):
        self.last_row_seen_time = self.get_clock().now()
        self.seen_row_this_pass = False
        self.next_row_hits = 0  # fresh count for THIS return pass
        self.set_state("FOLLOW_BACK", status)

    # -----------------------------
    # TURN_NEXT - row transition (in-place 180 deg toward the next row)
    # -----------------------------

    def turn_to_next_row(self):
        """Rotate in place back to the OUTBOUND heading to start the next row.

        Geometry: at the end of FOLLOW_BACK the robot is just past the
        start end of the finished row, heading along the RETURN direction,
        with the finished row on current_side and the next row on the
        opposite side (~row spacing minus follow offset away). Rotating
        180 deg in place toward the next row leaves the robot heading
        OUTBOUND with the next row on current_side - exactly the start
        configuration of a normal row sweep. In place = zero turning
        radius, so nothing in the corridor can be hit.

        The turn direction is FORCED toward the next row's side (away from
        the finished row) so the maneuver is predictable.
        """
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during TURN_NEXT: front={front:.2f}m")
            return

        if self.outbound_yaw is None or self.odom_yaw is None:
            self.stop_robot()
            self.publish_status("TURN_NEXT cannot proceed: no heading reference.")
            if self.elapsed_in_state() > self.turn_next_max_duration:
                self._begin_next_row("TURN_NEXT aborted blind - trying next row.")
            return

        err = self.normalize_angle(self.outbound_yaw - self.odom_yaw)

        if abs(err) < self.align_parallel_tol:
            self._begin_next_row(
                f"TURN_NEXT done (err={math.degrees(err):+.1f}deg). "
                f"Starting row {self.rows_completed + 1}."
            )
            return

        if self.elapsed_in_state() > self.turn_next_max_duration:
            self._begin_next_row(
                f"TURN_NEXT timeout (err={math.degrees(err):+.0f}deg). "
                f"Starting row {self.rows_completed + 1} anyway."
            )
            return

        # Forced direction: next row is on the OPPOSITE side of the
        # finished row. Finished row on the right -> next row on the left
        # -> rotate CCW (+), and vice versa.
        turn_dir = +1.0 if self.current_side == "right" else -1.0
        mag = min(self.align_max_angular,
                  max(self.min_align_angular, self.k_align * abs(err)))
        self.publish_cmd(0.0, turn_dir * mag)
        self.publish_status(
            f"TURN_NEXT | odom={math.degrees(self.odom_yaw):+.0f}deg "
            f"target={math.degrees(self.outbound_yaw):+.0f}deg "
            f"err={math.degrees(err):+.0f}deg | rot={turn_dir * mag:+.2f}"
        )

    def _begin_next_row(self, status: str):
        """Reset per-row state and start the outbound pass on the next row."""
        self.next_row_hits = 0
        self.seen_row_this_pass = False
        self.last_row_seen_time = self.get_clock().now()
        self.clear_start_x = None
        self.clear_start_y = None
        self.arc_start_yaw = None
        self.arc_last_yaw = None
        self.arc_accumulated_yaw = 0.0
        # current_side is UNCHANGED: after the in-place 180 the next row
        # sits on the same commanded side as the previous one did.
        self.set_state("FOLLOW_OUT", status)

    # -----------------------------
    # RETURN_HOME
    # -----------------------------

    def finish_or_return_home(self, reason: str):
        """Go to RETURN_HOME if enabled, otherwise finish immediately."""
        self.stop_robot()
        if self.return_home_enabled:
            if self.home_x is None or self.home_y is None:
                self.set_state(
                    "DONE",
                    reason + " Cannot return home because start odom was not saved."
                )
            else:
                self.set_state(
                    "RETURN_HOME",
                    reason + f" Returning to start x={self.home_x:+.2f}, y={self.home_y:+.2f}."
                )
        else:
            self.set_state("DONE", reason + " Mission complete.")

    def return_home(self):
        """Drive back to the odometry pose recorded at /sweep_start."""
        if self.home_x is None or self.home_y is None:
            self.set_state("DONE", "No home pose saved. Robot stopped.")
            self.stop_robot()
            return

        if self.odom_x is None or self.odom_y is None or self.odom_yaw is None:
            self.stop_robot()
            self.publish_status("RETURN_HOME waiting for odometry")
            return

        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(
                f"RETURN_HOME blocked: front obstacle {front:.2f}m. Robot stopped."
            )
            return

        dx = self.home_x - self.odom_x
        dy = self.home_y - self.odom_y
        dist = math.hypot(dx, dy)

        # First reach the saved x/y position.
        if dist > self.return_goal_tolerance:
            target_yaw = math.atan2(dy, dx)
            heading_error = self.normalize_angle(target_yaw - self.odom_yaw)

            # If facing far away from home, rotate first. Otherwise drive forward
            # with proportional heading correction. This avoids driving a big arc.
            if abs(heading_error) > self.return_heading_slowdown:
                linear = 0.0
                angular = math.copysign(
                    self.return_max_angular,
                    heading_error,
                )
                mode = "turning toward start"
            else:
                linear = min(self.return_linear_speed, max(0.03, 0.45 * dist))
                angular = max(
                    -self.return_max_angular,
                    min(self.return_max_angular, 1.4 * heading_error),
                )
                mode = "driving to start"

            self.publish_cmd(linear, angular)
            self.publish_status(
                f"RETURN_HOME {mode} | dist={dist:.2f}m "
                f"| heading_err={math.degrees(heading_error):+.0f}deg "
                f"| front={front:.2f}m"
            )
            if self.elapsed_in_state() > self.return_max_duration:
                self.set_state(
                    "DONE",
                    f"RETURN_HOME timeout after {self.return_max_duration:.0f}s. Robot stopped."
                )
                self.stop_robot()
            return

        # At x/y start. Rotate back to the original start heading if known.
        if self.home_yaw is not None:
            yaw_error = self.normalize_angle(self.home_yaw - self.odom_yaw)
            if abs(yaw_error) > self.return_yaw_tolerance:
                angular = max(
                    -self.return_max_angular,
                    min(self.return_max_angular, 1.2 * yaw_error),
                )
                if abs(angular) < self.min_align_angular:
                    angular = math.copysign(self.min_align_angular, yaw_error)
                self.publish_cmd(0.0, angular)
                self.publish_status(
                    f"RETURN_HOME at start, aligning yaw | "
                    f"yaw_err={math.degrees(yaw_error):+.0f}deg"
                )
                return

        self.stop_robot()
        self.set_state(
            "DONE",
            f"Returned to start. Final distance={dist:.2f}m. Mission complete."
        )

    # -----------------------------
    # State machine
    # -----------------------------

    def control_loop(self):
        if not self.started:
            self.stop_robot()
            return
        if self.latest_scan is None:
            self.stop_robot()
            self.publish_status("Waiting for /scan")
            return

        # Lazy heading anchor: set on first odom tick if it wasn't ready at
        # sweep_start. The robot hasn't moved during start_straight_duration,
        # so this is still a valid "parallel to row" snapshot.
        if self.outbound_yaw is None and self.odom_yaw is not None:
            self.outbound_yaw = self.odom_yaw
            self.return_target_yaw = self.normalize_angle(self.odom_yaw + math.pi)
            self.get_logger().info(
                f"Lazy anchor: outbound={math.degrees(self.outbound_yaw):+.1f}deg "
                f"return_target={math.degrees(self.return_target_yaw):+.1f}deg"
            )

        if (self.home_x is None or self.home_y is None) and self.odom_x is not None and self.odom_y is not None:
            self.home_x = self.odom_x
            self.home_y = self.odom_y
            self.home_yaw = self.odom_yaw
            self.get_logger().info(
                f"Lazy home saved: x={self.home_x:+.2f}, y={self.home_y:+.2f}, "
                f"yaw={math.degrees(self.home_yaw or 0.0):+.1f}deg"
            )

        if self.state == "FOLLOW_OUT":
            self.follow_side(self.current_side)
            abeam = self.last_tree_abeam_or_behind(self.current_side)
            lost = self.should_end_pass()
            if abeam or lost:
                trigger = "last tree abeam" if abeam else "row lost"
                self.clear_start_x = None
                self.clear_start_y = None
                self.arc_start_yaw = None
                self.arc_last_yaw = None
                self.arc_accumulated_yaw = 0.0
                self.seen_row_this_pass = False
                tgt = (math.degrees(self.return_target_yaw)
                       if self.return_target_yaw is not None else float("nan"))
                self.set_state(
                    "CLEAR_END",
                    f"End of row ({trigger}). Return heading={tgt:.0f}deg. "
                    f"Driving past last tree."
                )
                return

        elif self.state == "CLEAR_END":
            self.clear_straight("ARC_TURN", "CLEAR_END")

        elif self.state == "ARC_TURN":
            self.arc_turn()

        elif self.state == "ALIGN":
            self.align_to_row()

        elif self.state == "FOLLOW_BACK":
            self.follow_side(self.current_side)

            # While sweeping back, the NEXT row (if any) sits on the
            # opposite side of the robot. Count confident fits so the
            # decision at the end of the pass is based on a whole pass of
            # evidence, not one noisy scan.
            opp = self.opposite(self.current_side)
            opp_visible, opp_fit = self.row_visible(opp)
            if opp_visible and abs(opp_fit[1]) <= self.next_row_max_dist:
                self.next_row_hits += 1

            if self.should_end_pass():
                self.rows_completed += 1
                next_row_seen = self.next_row_hits >= self.next_row_min_hits

                if self.max_rows > 0:
                    # FIXED COUNT (selected): proceed to the next row after
                    # each one until exactly max_rows are done, regardless of
                    # perception. next_row_seen is logged but not gating.
                    proceed = self.rows_completed < self.max_rows
                    done_reason = f"reached requested {self.max_rows} rows"
                else:
                    # AUTO: only continue while a next row is actually seen.
                    proceed = next_row_seen
                    done_reason = (f"no next row ({self.next_row_hits} hits "
                                   f"< {self.next_row_min_hits})")

                if proceed:
                    self.clear_start_x = None
                    self.clear_start_y = None
                    seen_str = (f"next row on {opp.upper()} "
                                f"({self.next_row_hits} hits)" if next_row_seen
                                else "next row not yet confirmed by lidar")
                    self.set_state(
                        "CLEAR_NEXT",
                        f"Row {self.rows_completed} done; {seen_str}. "
                        f"Clearing row start."
                    )
                else:
                    self.finish_or_return_home(
                        f"Row {self.rows_completed} done, {done_reason}."
                    )
                return

        elif self.state == "CLEAR_NEXT":
            self.clear_straight("TURN_NEXT", "CLEAR_NEXT")

        elif self.state == "TURN_NEXT":
            self.turn_to_next_row()

        elif self.state == "RETURN_HOME":
            self.return_home()

        elif self.state == "DONE":
            self.stop_robot()
            self.started = False
            self.publish_status("Demo complete. Robot stopped.")
            return

        elif self.state == "STOPPED":
            self.stop_robot()
            return

        else:
            self.stop_robot()
            self.publish_status(f"Unknown state: {self.state}")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleRowFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()