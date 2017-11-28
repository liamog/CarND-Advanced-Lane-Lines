
class Config():

    # Lane width in pixels after perspective transform. Make sure to keep in
    # sync with transform.
    LANE_WIDTH_IN_PIXELS = (980 - 320)
    # Max number of frames to combine when searching
    # for lanes.
    SMOOTH_OVER_N_FRAMES = 2

    # Number of windows for the sliding window search
    SEARCH_WINDOWS = 9
    
    # Search window margin for both the sliding window and
    # fast fit search.
    MARGIN = 75

    #Minimum number of pixels in a window to recenter and fit to.
    MIN_PIXEL_PER_WINDOW = 75

    # If 5 frames in a row don't have a good lane match
    # reset to sliding window search
    MAX_REJECTED = 5

    # meters per pixel in y dimension 
    # Measured with output images from straight_line1.jpg warped with same
    # perspective. 3m Lane line is 60px long.
    YM_PER_PIX = 3 / 60
    
    # meters per pixel in x dimension
    # Calculated from the warped perspective transform used.
    XM_PER_PIX = 3.7 / LANE_WIDTH_IN_PIXELS

    # Y Range at to search for lines.
    WARPED_Y_RANGE = (0, 700)

    # Shape of the warped image.
    WARPED_SHAPE = (720, 1280)

    # Minimum smooth polynomial cofficients samples before 
    # switching to fast detection.
    MIN_SMOOTH_SAMPLES = 3
    
    # Maximum number of fit samples to smooth over.
    MAX_SMOOTH_SAMPLES = 5

    # Maximum standard deviation (in pixels) below which we
    # consider as parallel therefore a good lane detection. 
    PROBABLE_LANE_WIDTH_STDDEV = 40
    
    # Range of average lane widths to accept as a probable lane
    # Larger lanes more probable than narrower lanes.
    PROBABLE_LANE_AVERAGE_WIDTH_RANGE = (
        LANE_WIDTH_IN_PIXELS - 75, LANE_WIDTH_IN_PIXELS + 150)
