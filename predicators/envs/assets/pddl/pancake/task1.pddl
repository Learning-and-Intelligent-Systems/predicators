(define (problem pancake-v1)
	(:domain pancake)
	(:objects
    	arm - robot
    	pan - pan
    	oil - oil
    	spatula - locatable
    	mix - mix
    	wet_mix - wet_mix
    	pancake - pancake
    	water - water
    	bowl - bowl
    	plate - plate
    	orange - locatable
    	knife - locatable
    	bread - locatable
    	ketchup - locatable
    	countertop - location
    	cupboard1 - location
    	cupboard2 - location
    	cupboard3 - location
    	stove - stove
    	
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
	
	(on pan cupboard1)
	(on oil cupboard1)
	(on spatula cupboard1)
	(on mix cupboard2)
	(on water cupboard2)
	(on knife cupboard2)
	(on bowl cupboard3)
	(on plate cupboard2)
	(on orange countertop)
	(on ketchup countertop)
	(on bread countertop)
	
	(arm-empty)
	


	
)

(:goal 
	(and(exist pancake))
	
    )
)
