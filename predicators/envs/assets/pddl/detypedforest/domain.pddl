(define (domain forest)

(:predicates
  (loc ?loc)
  (at ?loc)
  (isNotWater ?loc)
  (isHill ?loc)
  (isNotHill ?loc)
  (adjacent ?loc1 ?loc2)
  (onTrail ?from ?to)
)

(:action walk
  :parameters (?from ?to)
  :precondition (and
    (loc ?from)
    (loc ?to)
    (isNotHill ?to)
    (at ?from)
    (adjacent ?from ?to)
    (isNotWater ?from))
  :effect (and (at ?to) (not (at ?from)))
)

(:action climb
  :parameters (?from ?to)
  :precondition (and
    (loc ?from)
    (loc ?to)
    (isHill ?to)
    (at ?from)
    (adjacent ?from ?to)
    (isNotWater ?from))
  :effect (and (at ?to) (not (at ?from)))
)


)