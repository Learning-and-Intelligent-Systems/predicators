(define (domain pancake)
  
  (:requirements :typing) 
  
  (:types  

		bot location

  )

  (:predicates
        
        (isStove ?stove - location)
        (isArm ?arm - object)
        (isBowl ?bowl - object)
        (isPan ?pan - object)
        (isOil ?oil - object)
        (isWater ?water - object)
        (isMix ?mix - object)
        (isWet_Mix ?wet_mix - object)
        (isPancake ?pancake - object)
        (at ?arm - bot ?loc - location)
		(on ?obj - object ?loc - location)
		(holding ?arm - bot ?obj - object)
    (arm-empty)
    (path ?location1 - location ?location2 - location)
    (in ?item - object ?obj - object)
    (exist ?item - object)
    
  )


  (:action mixing
    :parameters
     (?arm - bot
     ?mix - object
     ?wet_mix - object
     ?water - object
     ?bowl - object
      ?loc - location)
    :precondition
    (and
        (exist ?arm)
        (isMix ?mix)
        (isWater ?water)
        (at ?arm ?loc)
        (on ?bowl ?loc)
        (isBowl ?bowl)
        (in ?water ?bowl)
        (in ?mix ?bowl)
    )
    :effect
    (and
        (exist ?wet_mix)
        (in ?wet_mix ?bowl)
    
    )
  
  )

  (:action cook
    :parameters
     (?wet_mix - object
     ?oil - object
     ?pan - object
     ?stove - location
     ?pancake - object
     
     )

    :precondition
    (and
        (exist ?wet_mix)
        (isWet_Mix ?wet_mix)
        (isOil ?oil)
        (on ?pan ?stove)
        (isPan ?pan)
        (isStove ?stove)
        (in ?wet_mix ?pan)
        (in ?oil ?pan)
    )
    :effect
    (and
        (exist ?pancake)
        (isPancake ?pancake)
        (in ?pancake ?pan)
        (not(exist ?wet_mix))
    
    )
  
  )

  (:action pour
    :parameters
     (?arm - bot
      ?pan - object
      ?bowl - object
      ?item - object
      ?loc - location)
    :precondition
     (and
        (at ?arm ?loc)
        (on ?pan ?loc)
        (holding ?arm ?bowl)
        (in ?item ?bowl)
     
     )
    :effect
     (and
        (in ?item ?pan)
     
     )
    
  )


  (:action pick-up
    :parameters
     (?arm - bot
      ?obj - object
      ?loc - location)
    :precondition
     (and 
        (at ?arm ?loc) 
        (on ?obj ?loc)
        (arm-empty)
      )
    :effect
     (and 
        (not (on ?obj ?loc))
        (holding ?arm ?obj)
        (not (arm-empty))
     )
  )


  (:action drop
    :parameters
     (?arm - bot
      ?obj - object
      ?loc - location)
    :precondition
     (and 
        (at ?arm ?loc)
        (holding ?arm ?obj)
      )
    :effect
     (and 
        (arm-empty)
        (on ?obj ?loc)
        (not (holding ?arm ?obj))
     )
  )


  (:action move
    :parameters
     (?arm - bot
      ?from - location
      ?to - location)
    :precondition
     (and 
      (at ?arm ?from) 
      (path ?from ?to)
     )
    :effect
     (and 
      (not (at ?arm ?from))
      (at ?arm ?to)
     )
  )


  (:action place
    :parameters
        (
        ?arm - bot
        ?obj - object
        ?item - object
        ?loc - location
        )
  
    :precondition
        (and
            (on ?obj ?loc)
            (holding ?arm ?item)
            (at ?arm ?loc)
            
               
        )
  
    :effect
    (and
        (arm-empty)
        (in ?item ?obj)
        (on ?obj ?loc)
        (not(holding ?arm ?item))
        
    )
        
    
  )

    
)  
