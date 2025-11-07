(define (domain initial-domain)
 (:requirements :strips :typing :negative-preconditions)
 (:types robot location sensor victim)
 (:predicates (at_ ?r - robot ?l - location) (adjacent ?l1 - location ?l2 - location) (has_sensor ?r - robot ?s - sensor) (sensor_active ?s - sensor) (explored ?l - location) (blocked ?p - location) (dangerous ?a - location) (stable ?a - location) (victim_at ?v - victim ?l - location) (detected ?v - victim) (rescued ?v - victim) (reported ?v - victim) (communication_available ?l - location) (is_thermal ?s - sensor) (is_lidar ?s - sensor))
 (:action move
  :parameters ( ?r - robot ?from_loc - location ?to_loc - location)
  :precondition (and (at_ ?r ?from_loc) (adjacent ?from_loc ?to_loc) (not (blocked ?to_loc)))
  :effect (and (not (at_ ?r ?from_loc)) (at_ ?r ?to_loc) (explored ?to_loc)))
 (:action activate_sensor
  :parameters ( ?r - robot ?s - sensor)
  :precondition (and (has_sensor ?r ?s) (not (sensor_active ?s)))
  :effect (and (sensor_active ?s)))
 (:action deactivate_sensor
  :parameters ( ?r - robot ?s - sensor)
  :precondition (and (has_sensor ?r ?s) (sensor_active ?s))
  :effect (and (not (sensor_active ?s))))
 (:action detect_victim
  :parameters ( ?r - robot ?v - victim ?l - location ?s - sensor)
  :precondition (and (at_ ?r ?l) (victim_at ?v ?l) (has_sensor ?r ?s) (sensor_active ?s) (is_thermal ?s))
  :effect (and (detected ?v)))
 (:action scan_area
  :parameters ( ?r - robot ?a - location ?s - sensor)
  :precondition (and (at_ ?r ?a) (has_sensor ?r ?s) (sensor_active ?s) (is_lidar ?s))
  :effect (and (explored ?a)))
 (:action report_victim
  :parameters ( ?r - robot ?v - victim ?l - location)
  :precondition (and (at_ ?r ?l) (detected ?v) (not (reported ?v)) (communication_available ?l))
  :effect (and (reported ?v)))
 (:action clear_path
  :parameters ( ?r - robot ?p - location ?a - location)
  :precondition (and (at_ ?r ?a) (adjacent ?a ?p) (blocked ?p))
  :effect (and (not (blocked ?p))))
 (:action rescue_victim
  :parameters ( ?r - robot ?v - victim ?a - location)
  :precondition (and (at_ ?r ?a) (victim_at ?v ?a) (detected ?v) (reported ?v) (stable ?a))
  :effect (and (rescued ?v)))
)
