
class Config():
    # Max number of frames to combine when searching
    # for lanes.
    SMOOTH_OVER_N_FRAMES = 3
    
    # Number of windows for the sliding window search
    SEARCH_WINDOWS = 9
    
    # Search window margin for both the sliding window and
    # fast fit search.
    MARGIN = 120
    # If 5 frames in a row don't have a good lane match
    # reset to sliding window search
    MAX_REJECTED = 5

    # meters per pixel in y dimension
    YM_PER_PIX = 35 / 720  
    
    # meters per pixel in x dimension
    # Calculated from the warped perspective transform used.
    XM_PER_PIX = 3.7 / (1080 - 230)  

    # Minimum smooth polynomial cofficients samples before 
    # switching to fast detection.
    MIN_SMOOTH_SAMPLES = 3
    
    # Maximum number of fit samples to smooth over.
    MAX_SMOOTH_SAMPLES = 8

    # Maximum standard deviation (in pixels) below which we
    # consider as parallel therefor a good lane detection. 
    PROBABLE_LANE_WIDTH_STDDEV = 25
    
    # Range of average lane widths to accept as a probable lane
    PROBABLE_LANE_AVERAGE_WIDTH_RANGE = (750, 950)
