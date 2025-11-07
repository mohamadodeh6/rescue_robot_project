import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import defaultdict
import time

LOC_COORDS = {
    "entrance":       (0.0, 0.0),
    "path_entrance":  (1.5, 0.0),
    "corridor1":      (3.0, 0.0),
    "path_corr1":     (3.0, 2.0),
    "room1":          (3.0, 4.5),
    "path_corr2":     (4.5, 0.0),
    "corridor2":      (6.0, 0.0),
    "path1_2":        (6.0, -2.0),
    "room2":          (6.0, -4.5),
    "path2_3":        (7.5, 0.0),
    "room3":          (9.0, 0.0),
}

ADJACENCY = {
    ("entrance", "path_entrance"),
    ("path_entrance", "corridor1"),
    ("corridor1", "path_corr1"),
    ("path_corr1", "room1"),
    ("corridor1", "path_corr2"),
    ("path_corr2", "corridor2"),
    ("corridor2", "path1_2"),
    ("path1_2", "room2"),
    ("corridor2", "path2_3"),
    ("path2_3", "room3"),
}
ADJACENCY |= {(b, a) for (a, b) in list(ADJACENCY)}

COMM_ZONES = {"entrance", "corridor1"}
STABLE_LOCS = {"entrance", "corridor1", "room1", "room2", "corridor2"}
DANGEROUS_LOCS = {"room3"}
VICTIM_LOCATIONS = {"victim1": "room1", "victim2": "room3"}


class SearchRescueSimulator:
    """
    Simulates the UPF search and rescue mission with matplotlib visualization.
    """

    def __init__(self):
        # UPF state mirrors
        self.has_sensor = {"thermal_cam": True, "lidar_scanner": True}
        self.is_thermal = {"thermal_cam": True}
        self.is_lidar = {"lidar_scanner": True}
        self.sensor_active = {"thermal_cam": False, "lidar_scanner": False}
        self.blocked = {"path_corr2": True}
        self.explored = {loc: False for loc in LOC_COORDS.keys()}
        self.victim_at = VICTIM_LOCATIONS.copy()
        self.detected = {"victim1": False, "victim2": False}
        self.reported = {"victim1": False, "victim2": False}
        self.rescued = {"victim1": False, "victim2": False}
        self.current_location = "entrance"

        # Trajectory tracking
        self.trajectory = [LOC_COORDS["entrance"]]
        self.action_log = []

        # Generate plan
        self.plan = self._generate_plan()

        # Setup plot
        self.fig, (self.ax_map, self.ax_state) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]}
        )
        self.setup_plot()

    def _generate_plan(self):
        """Generate mission plan (same logic as Webots controller)."""
        return [
            ("activate_sensor", "lidar_scanner"),
            ("activate_sensor", "thermal_cam"),
            ("move", "entrance", "path_entrance"),
            ("move", "path_entrance", "corridor1"),
            ("move", "corridor1", "path_corr1"),
            ("move", "path_corr1", "room1"),
            ("scan_area", "room1", "lidar_scanner"),
            ("detect_victim", "victim1", "room1", "thermal_cam"),
            ("move", "room1", "path_corr1"),
            ("move", "path_corr1", "corridor1"),
            ("report_victim", "victim1", "corridor1"),
            ("move", "corridor1", "path_corr1"),
            ("move", "path_corr1", "room1"),
            ("rescue_victim", "victim1", "room1"),
            ("move", "room1", "path_corr1"),
            ("move", "path_corr1", "corridor1"),
            ("clear_path", "path_corr2", "corridor1"),
            ("move", "corridor1", "path_corr2"),
            ("move", "path_corr2", "corridor2"),
            ("move", "corridor2", "path2_3"),
            ("move", "path2_3", "room3"),
            ("scan_area", "room3", "lidar_scanner"),
            ("detect_victim", "victim2", "room3", "thermal_cam"),
            ("move", "room3", "path2_3"),
            ("move", "path2_3", "corridor2"),
            ("move", "corridor2", "path_corr2"),
            ("move", "path_corr2", "corridor1"),
            ("report_victim", "victim2", "corridor1"),
            ("move", "corridor1", "path_entrance"),
            ("move", "path_entrance", "entrance"),
        ]

    def setup_plot(self):
        """Initialize the matplotlib plot."""
        self.ax_map.set_xlim(-1, 10)
        self.ax_map.set_ylim(-6, 6)
        self.ax_map.set_aspect('equal')
        self.ax_map.set_title('Search & Rescue Mission - Map View', fontsize=14, fontweight='bold')
        self.ax_map.set_xlabel('X (meters)')
        self.ax_map.set_ylabel('Y (meters)')
        self.ax_map.grid(True, alpha=0.3)

        # Draw adjacency graph (edges)
        for (loc1, loc2) in ADJACENCY:
            if loc1 < loc2:  # Draw each edge once
                x_vals = [LOC_COORDS[loc1][0], LOC_COORDS[loc2][0]]
                y_vals = [LOC_COORDS[loc1][1], LOC_COORDS[loc2][1]]
                self.ax_map.plot(x_vals, y_vals, 'k-', alpha=0.3, linewidth=1)

        # Draw location markers
        for loc, (x, y) in LOC_COORDS.items():
            color = self._get_location_color(loc)
            marker = 's' if loc.startswith('room') else 'o'
            size = 150 if loc.startswith('room') else 100
            self.ax_map.scatter(x, y, c=color, s=size, marker=marker,
                              edgecolors='black', linewidths=1.5, alpha=0.7, zorder=2)
            self.ax_map.text(x, y-0.3, loc.replace('_', '\n'), fontsize=7,
                           ha='center', va='top')

        # Draw victims
        for victim, loc in VICTIM_LOCATIONS.items():
            x, y = LOC_COORDS[loc]
            self.ax_map.scatter(x, y, c='red', s=200, marker='X',
                              edgecolors='darkred', linewidths=2, zorder=3)
            self.ax_map.text(x+0.3, y+0.3, victim, fontsize=8,
                           color='darkred', fontweight='bold')

        # Robot position (will be updated)
        self.robot_marker = self.ax_map.scatter([], [], c='blue', s=300,
                                                marker='D', edgecolors='darkblue',
                                                linewidths=2, zorder=5)

        # Trajectory line
        self.traj_line, = self.ax_map.plot([], [], 'b--', linewidth=2, alpha=0.6)

        # Sensor range circles (initially hidden)
        self.lidar_circle = plt.Circle((0, 0), 1.5, color='cyan',
                                       fill=False, linewidth=2, linestyle='--',
                                       alpha=0, zorder=1)
        self.thermal_circle = plt.Circle((0, 0), 1.0, color='red',
                                         fill=False, linewidth=2, linestyle='--',
                                         alpha=0, zorder=1)
        self.ax_map.add_patch(self.lidar_circle)
        self.ax_map.add_patch(self.thermal_circle)

        # Blocked path indicator
        if self.blocked.get("path_corr2", False):
            bx, by = LOC_COORDS["path_corr2"]
            self.blocked_marker = self.ax_map.scatter(bx, by, c='orange',
                                                     s=400, marker='X',
                                                     edgecolors='darkorange',
                                                     linewidths=3, alpha=0.8, zorder=4)
        else:
            self.blocked_marker = None

        # State display (right panel)
        self.ax_state.axis('off')
        self.state_text = self.ax_state.text(0.05, 0.95, '', fontsize=10,
                                            verticalalignment='top',
                                            family='monospace')

        plt.tight_layout()

    def _get_location_color(self, loc):
        """Determine location marker color based on properties."""
        if loc in COMM_ZONES:
            return 'green'
        elif loc in STABLE_LOCS:
            return 'lightblue'
        elif loc in DANGEROUS_LOCS:
            return 'red'
        else:
            return 'lightgray'

    def _update_state_display(self):
        """Update the state information panel."""
        state_str = "═══ MISSION STATE ═══\n\n"
        state_str += f"Current Location:\n  {self.current_location}\n\n"
        state_str += f"Sensors Active:\n"
        state_str += f"  Lidar: {'ON' if self.sensor_active['lidar_scanner'] else 'OFF'}\n"
        state_str += f"  Thermal: {'ON' if self.sensor_active['thermal_cam'] else 'OFF'}\n\n"
        state_str += f"Victims:\n"
        state_str += f"  victim1:\n"
        state_str += f"    Detected: {'✓' if self.detected['victim1'] else '✗'}\n"
        state_str += f"    Reported: {'✓' if self.reported['victim1'] else '✗'}\n"
        state_str += f"    Rescued:  {'✓' if self.rescued['victim1'] else '✗'}\n"
        state_str += f"  victim2:\n"
        state_str += f"    Detected: {'✓' if self.detected['victim2'] else '✗'}\n"
        state_str += f"    Reported: {'✓' if self.reported['victim2'] else '✗'}\n"
        state_str += f"    Rescued:  {'✓' if self.rescued['victim2'] else '✗'}\n\n"
        state_str += f"Environment:\n"
        state_str += f"  Explored: {sum(self.explored.values())}/11\n"
        state_str += f"  Blocked: {'path_corr2' if self.blocked.get('path_corr2') else 'None'}\n\n"

        # Goals status
        state_str += "═══ GOALS ═══\n"
        state_str += f"rescued(victim1):  {'✓' if self.rescued['victim1'] else '✗'}\n"
        state_str += f"reported(victim2): {'✓' if self.reported['victim2'] else '✗'}\n"
        state_str += f"at(robot, entrance): {'✓' if self.current_location == 'entrance' else '✗'}\n"

        self.state_text.set_text(state_str)

    def execute_action(self, action_tuple):
        """Execute a single action with validation."""
        action_name = action_tuple[0]
        args = action_tuple[1:]

        try:
            if action_name == "activate_sensor":
                self._action_activate_sensor(*args)
            elif action_name == "move":
                self._action_move(*args)
            elif action_name == "scan_area":
                self._action_scan_area(*args)
            elif action_name == "detect_victim":
                self._action_detect_victim(*args)
            elif action_name == "report_victim":
                self._action_report_victim(*args)
            elif action_name == "clear_path":
                self._action_clear_path(*args)
            elif action_name == "rescue_victim":
                self._action_rescue_victim(*args)

            log_entry = f"✓ {action_name}{args}"
            self.action_log.append(log_entry)
            print(log_entry)

        except AssertionError as e:
            log_entry = f"✗ {action_name}{args} FAILED: {e}"
            self.action_log.append(log_entry)
            print(log_entry)
            raise

    def _action_activate_sensor(self, sensor_name):
        assert self.has_sensor.get(sensor_name, False)
        assert not self.sensor_active.get(sensor_name, False)
        self.sensor_active[sensor_name] = True

    def _action_move(self, from_loc, to_loc):
        assert self.current_location == from_loc
        assert (from_loc, to_loc) in ADJACENCY
        assert not self.blocked.get(to_loc, False)

        self.current_location = to_loc
        self.trajectory.append(LOC_COORDS[to_loc])
        self.explored[to_loc] = True

    def _action_scan_area(self, area, sensor_name):
        assert self.current_location == area
        assert self.sensor_active.get(sensor_name, False)
        assert self.is_lidar.get(sensor_name, False)
        self.explored[area] = True

    def _action_detect_victim(self, victim, location, sensor_name):
        assert self.current_location == location
        assert self.victim_at[victim] == location
        assert self.sensor_active.get(sensor_name, False)
        assert self.is_thermal.get(sensor_name, False)
        self.detected[victim] = True

    def _action_report_victim(self, victim, location):
        assert self.current_location == location
        assert self.detected[victim]
        assert not self.reported[victim]
        assert location in COMM_ZONES
        self.reported[victim] = True

    def _action_clear_path(self, blocked_loc, adjacent_loc):
        assert self.current_location == adjacent_loc
        assert (adjacent_loc, blocked_loc) in ADJACENCY
        assert self.blocked.get(blocked_loc, False)
        self.blocked[blocked_loc] = False

    def _action_rescue_victim(self, victim, location):
        assert self.current_location == location
        assert self.victim_at[victim] == location
        assert self.detected[victim]
        assert self.reported[victim]
        assert location in STABLE_LOCS
        self.rescued[victim] = True

    def _update_visualization(self, frame):
        """Update plot visuals without executing actions (for static mode)."""
        # Update robot position
        x, y = LOC_COORDS[self.current_location]
        self.robot_marker.set_offsets([[x, y]])

        # Update trajectory
        traj_x = [p[0] for p in self.trajectory]
        traj_y = [p[1] for p in self.trajectory]
        self.traj_line.set_data(traj_x, traj_y)

        # Update sensor range indicators
        if self.sensor_active.get('lidar_scanner', False):
            self.lidar_circle.set_center((x, y))
            self.lidar_circle.set_alpha(0.3)
        else:
            self.lidar_circle.set_alpha(0)

        if self.sensor_active.get('thermal_cam', False):
            self.thermal_circle.set_center((x, y))
            self.thermal_circle.set_alpha(0.3)
        else:
            self.thermal_circle.set_alpha(0)

        # Update blocked indicator
        if not self.blocked.get("path_corr2", False) and self.blocked_marker:
            self.blocked_marker.remove()
            self.blocked_marker = None

        # Highlight explored locations
        for loc, explored in self.explored.items():
            if explored:
                lx, ly = LOC_COORDS[loc]
                self.ax_map.add_patch(plt.Circle((lx, ly), 0.15,
                                                color='yellow', alpha=0.3, zorder=0))

        # Update state panel
        self._update_state_display()

        # Update title with current action
        if frame < len(self.plan):
            action_str = f"{self.plan[frame][0]}{self.plan[frame][1:]}"
            self.ax_map.set_title(f'Action {frame+1}/{len(self.plan)}: {action_str}',
                                 fontsize=12, fontweight='bold')

    def update_plot(self, frame):
        """Animation update function (executes action + updates visuals)."""
        if frame >= len(self.plan):
            return

        # Execute action
        self.execute_action(self.plan[frame])

        # Update visuals
        self._update_visualization(frame)

    def run_simulation(self, interval=800):
        """Run the animated simulation."""
        print("\n" + "="*70)
        print(" SEARCH & RESCUE MISSION SIMULATION")
        print("="*70 + "\n")

        anim = FuncAnimation(self.fig, self.update_plot,
                           frames=len(self.plan),
                           interval=interval,
                           repeat=False)
        plt.show()

        self.print_mission_report()

    def run_static(self):
        """Run simulation without animation (instant execution)."""
        print("\n" + "="*70)
        print(" SEARCH & RESCUE MISSION SIMULATION (STATIC)")
        print("="*70 + "\n")

        for i, action in enumerate(self.plan):
            self.execute_action(action)
            self._update_visualization(i)

        plt.show()
        self.print_mission_report()

    def print_mission_report(self):
        """Print final mission report."""
        print("\n" + "="*70)
        print(" MISSION REPORT")
        print("="*70)
        print(f"\nTotal Actions Executed: {len(self.action_log)}")
        print(f"Final Location: {self.current_location}")
        print(f"Locations Explored: {sum(self.explored.values())}/11")

        print("\n--- Victim Status ---")
        for victim in ["victim1", "victim2"]:
            print(f"{victim}:")
            print(f"  Detected: {'✓' if self.detected[victim] else '✗'}")
            print(f"  Reported: {'✓' if self.reported[victim] else '✗'}")
            print(f"  Rescued:  {'✓' if self.rescued[victim] else '✗'}")

        print("\n--- Goal Verification ---")
        goals = [
            ("rescued(victim1)", self.rescued["victim1"]),
            ("reported(victim2)", self.reported["victim2"]),
            ("at(rescue_bot, entrance)", self.current_location == "entrance")
        ]
        for goal_name, achieved in goals:
            status = "✓ SATISFIED" if achieved else "✗ FAILED"
            print(f"{goal_name}: {status}")

        all_achieved = all(g[1] for g in goals)
        print(f"\nOverall Result: {'SUCCESS' if all_achieved else 'FAILURE'}")
        print("="*70 + "\n")


if __name__ == "__main__":
    import sys

    try:

        simulator = SearchRescueSimulator()
        print("\nExecuting static simulation...\n")
        simulator.run_static()

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)