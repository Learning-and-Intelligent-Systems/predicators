(define (domain delivery)

    (:predicates 
        (loc ?loc)
        (paper ?paper)
        (at ?loc)
        (isHomeBase ?loc)
        (satisfied ?loc)
        (wantsPaper ?loc)
        (safe ?loc)
        (unpacked ?paper)
        (carrying ?paper)
    )
    
    (:action pick-up
        :parameters (?paper ?loc)
        :precondition (and
            (paper ?paper)
            (loc ?loc)
            (at ?loc)
            (isHomeBase ?loc)
            (unpacked ?paper)
        )
        :effect (and
            (not (unpacked ?paper))
            (carrying ?paper)
        )
    )
    
    (:action move
        :parameters (?from ?to)
        :precondition (and
            (loc ?from)
            (loc ?to)
            (at ?from)
            (safe ?from)
        )
        :effect (and
            (not (at ?from))
            (at ?to)
        )
    )
    
    (:action deliver
        :parameters (?paper ?loc)
        :precondition (and
            (paper ?paper)
            (loc ?loc)
            (at ?loc)
            (carrying ?paper)
        )
        :effect (and
            (not (carrying ?paper))
            (not (wantsPaper ?loc))
            (satisfied ?loc)
        )
    )
    
)