def print_termination(termination_cond):
    # Unpack the termination condition
    test_termination_cond = termination_cond[0:4]
    join_termination_cond = termination_cond[4:6]
    obs_termination_cond  = termination_cond[6:10]
        
    test_is_reached, test_is_outside, test_is_grounded, test_is_failed_nav = test_termination_cond
    is_near_collision, is_collision = join_termination_cond
    obs_is_reached, obs_is_outside, obs_is_grounded, obs_is_failed_nav = obs_termination_cond
    
    op_status = ''
    done_status = ''
    
    if test_is_reached:
        done_status += '|Ship under test reaches final destination|'
        
    if test_is_outside:
        done_status += '|Ship under test goes outside map horizon|'
        
    if test_is_grounded:
        done_status += '|Ship under test experiences grounding|'
        
    if test_is_failed_nav:
        op_status += '|Ship under test suffers navigational failure|'
        
    if obs_is_reached:
        op_status += '|Obstacle ship reaches final destination|'
        
    if obs_is_outside:
        op_status += '|Obstacle ship goes outisde map horizon|'
        
    if obs_is_grounded:
        done_status += '|Obstacle ship experiences grounding|'
        
    if obs_is_failed_nav:
        done_status += '|Obstacle ship suffers navigational failure|'
        
    if is_near_collision:
        op_status += '|Near-collision warning activated!|'
        
    if is_collision:
        done_status += '|SHIP COLLISION!|'
        
    # Print status
    print('Operational status:', op_status)
    print('Termination status:', done_status)