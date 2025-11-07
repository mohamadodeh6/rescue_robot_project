(define (problem initial-problem)
 (:domain initial-domain)
 (:objects
   rescue_bot - robot
   room1 room2 room3 corridor1 corridor2 entrance path1_2 path2_3 path_corr1 path_corr2 path_entrance - location
   thermal_cam lidar_scanner - sensor
   victim1 victim2 - victim
 )
 (:init (at_ rescue_bot entrance) (has_sensor rescue_bot thermal_cam) (has_sensor rescue_bot lidar_scanner) (is_thermal thermal_cam) (is_lidar lidar_scanner) (adjacent entrance path_entrance) (adjacent path_entrance entrance) (adjacent path_entrance corridor1) (adjacent corridor1 path_entrance) (adjacent corridor1 path_corr1) (adjacent path_corr1 corridor1) (adjacent path_corr1 room1) (adjacent room1 path_corr1) (adjacent corridor1 path_corr2) (adjacent path_corr2 corridor1) (adjacent path_corr2 corridor2) (adjacent corridor2 path_corr2) (adjacent corridor2 path1_2) (adjacent path1_2 corridor2) (adjacent path1_2 room2) (adjacent room2 path1_2) (adjacent corridor2 path2_3) (adjacent path2_3 corridor2) (adjacent path2_3 room3) (adjacent room3 path2_3) (blocked path_corr2) (dangerous room3) (stable entrance) (stable corridor1) (stable room1) (stable room2) (stable corridor2) (victim_at victim1 room1) (victim_at victim2 room3) (communication_available entrance) (communication_available corridor1))
 (:goal (and (rescued victim1) (reported victim2) (at_ rescue_bot entrance)))
)
