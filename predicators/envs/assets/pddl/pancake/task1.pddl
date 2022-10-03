(define (problem pancake-v1)
	(:domain pancake)
	(:objects
    	arm - bot
    	pan - object
    	oil - object
    	spatula - object
    	mix - object
    	wet_mix - object
    	pancake - object
    	water - object
    	bowl - object
    	plate - object
    	orange - object
    	knife - object
    	bread - object
    	ketchup - object
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

	(exist arm)
	(exist pan)
	
    (isStove stove)
    
    (isBowl bowl)
    (isPan pan)
    (isArm arm)
    (isOil oil)
    (isWater water)
    (isMix mix)
    (isWet_Mix wet_mix)
	
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
	(and(isPancake pancake))
	
    )
)
