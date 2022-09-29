(define (domain pancake)
  
  (:requirements :typing) 
  
  (:types         
    location locatable - object
        countertop cupboard stove - location
		bot pan oil water mix spatula pancake bowl plate orange knife bread ketchup wet_mix - locatable
    robot - bot
  )

  (:predicates
  
        (at ?arm - bot ?loc - location)
		(on ?obj - locatable ?loc - location)
		(holding ?arm - bot ?obj - locatable)
    (arm-empty)
    (path ?location1 - location ?location2 - location)
    (in ?item - locatable ?obj - locatable)
    (exist ?item - locatable)
    
  )


  (:action mixing
    :parameters
     (?arm - bot
     ?mix - mix
     ?wet_mix - wet_mix
     ?water - water
     ?bowl - bowl
      ?loc - location)
    :precondition
    (and
        (exist ?mix)
        (exist ?water)
        (at ?arm ?loc)
        (on ?bowl ?loc)
        (in ?water ?bowl)
        (in ?mix ?bowl)
    )
    :effect
    (and
        (exist ?wet_mix)
        (in ?wet_mix ?bowl)
        (not(exist ?mix))
        (not(exist ?water))
    
    )
  
  )

  (:action cook
    :parameters
     (?wet_mix - wet_mix
     ?oil - oil
     ?pan - pan
     ?stove - stove
     ?pancake - pancake
     
     )

    :precondition
    (and
        (exist ?wet_mix)
        (exist ?oil)
        (on ?pan ?stove)
        (in ?wet_mix ?pan)
        (in ?oil ?pan)
    )
    :effect
    (and
        (exist ?pancake)
        (in ?pancake ?pan)
        (not(exist ?wet_mix))
        (not(exist ?oil))
    
    )
  
  )

  (:action pour
    :parameters
     (?arm - bot
      ?obj - locatable
      ?obj_2 - locatable
      ?item - locatable
      ?loc - location)
    :precondition
     (and
        (at ?arm ?loc)
        (on ?obj ?loc)
        (holding ?arm ?obj_2)
        (in ?item ?obj_2)
     
     )
    :effect
     (and
        (in ?item ?obj)
     
     )
    
  )


  (:action pick-up
    :parameters
     (?arm - bot
      ?obj - locatable
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
      ?obj - locatable
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
        ?obj - locatable
        ?item - locatable
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




    
