(define (problem pancake-v2)
	(:domain pancake)
	(:objects
    	arm - robot
    	pan - locatable
    	oil - insertable
    	spatula - insertable
    	mix - insertable
    	wet_mix - insertable
    	pancake - insertable
    	water - insertable
    	bowl - container
    	plate - container
    	orange - insertable
    	knife - insertable
    	bread - insertable
    	ketchup - insertable
    	countertop - location
    	cupboard1 - location
    	cupboard2 - location
    	cupboard3 - location
    	stove - location
    	
	)
	
(:init
	(path countertop cupboard1)
	(path cupboard1 countertop)
	(path countertop cupboard2)
	(path cupboard2 countertop)
	(path countertop cupboard3)
	(path cupboard3 countertop)
	(path countertop countertop)
	(path countertop stove)
	(path stove countertop)
	(exist oil)
	(exist water)
	(exist mix)
	
	(at arm countertop)
	
	(on pan countertop)
	(on oil cupboard2)
	(on spatula cupboard1)
	(on mix cupboard2)
	(on water cupboard2)
	(on knife cupboard2)
	(on bowl cupboard3)
	(on plate cupboard2)
	(on orange cupboard1)
	(on ketchup countertop)
	(on bread countertop)
	
	(arm-empty)
	


	
)

(:goal 
	(and(exist pancake))
	
    )
)
