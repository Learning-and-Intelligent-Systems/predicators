(define (domain forest)
  (:requirements :strips :typing)
  (:types loc)

(:predicates
  (at ?loc - loc)
  (isNotWater ?loc - loc)
  (isHill ?loc - loc)
  (isNotHill ?loc - loc)
  (adjacent ?loc1 - loc ?loc2 - loc)
  (onTrail ?from - loc ?to - loc)
)

(:action walk
  :parameters (?from - loc ?to - loc)
  :precondition (and
    (isNotHill ?to)
    (at ?from)
    (adjacent ?from ?to)
    (isNotWater ?from))
  :effect (and (at ?to) (not (at ?from)))
)

(:action climb
  :parameters (?from - loc ?to - loc)
  :precondition (and
    (isHill ?to)
    (at ?from)
    (adjacent ?from ?to)
    (isNotWater ?from))
  :effect (and (at ?to) (not (at ?from)))
)


)