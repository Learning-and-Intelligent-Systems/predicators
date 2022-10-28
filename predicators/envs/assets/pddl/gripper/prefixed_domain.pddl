(define (domain pregripper)
   (:predicates (preroom ?r)
		(preball ?b)
		(pregripper ?g)
		(preat-robby ?r)
		(preat ?b ?r)
		(prefree ?g)
		(precarry ?o ?g))

   (:action move
       :parameters  (?from ?to)
       :precondition (and  (preroom ?from) (preroom ?to) (preat-robby ?from))
       :effect (and  (preat-robby ?to)
		     (not (preat-robby ?from))))



   (:action pick
       :parameters (?obj ?preroom ?pregripper)
       :precondition  (and  (preball ?obj) (preroom ?preroom) (pregripper ?pregripper)
			    (preat ?obj ?preroom) (preat-robby ?preroom) (prefree ?pregripper))
       :effect (and (precarry ?obj ?pregripper)
		    (not (preat ?obj ?preroom)) 
		    (not (prefree ?pregripper))))


   (:action drop
       :parameters  (?obj  ?preroom ?pregripper)
       :precondition  (and  (preball ?obj) (preroom ?preroom) (pregripper ?pregripper)
			    (precarry ?obj ?pregripper) (preat-robby ?preroom))
       :effect (and (preat ?obj ?preroom)
		    (prefree ?pregripper)
		    (not (precarry ?obj ?pregripper)))))