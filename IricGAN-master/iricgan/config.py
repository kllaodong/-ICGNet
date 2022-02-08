class Config:
    
    model={
        'bald':'Bald_linear_boundary.npy',
        'fear':'emotion_boundary_style_fear_neutral.npy',
        'happy':'emotion_boundary_style_happy_neutral.npy',
        'sad':'emotion_boundary_style_sad_neutral.npy',
        'surprise':'emotion_boundary_style_surprise_neutral.npy',
        'angry':'generate_angry.npy',
        'mustache':'Mustache_linear_boundary.npy',
        'old':'linear_boundary_age.npy',
        'gender':'linear_boundary_female.npy',        
    }
    
    strength={
        'bald':1,
        'fear':-1/7,
        'happy':-1/4,
        'sad':-1/4,
        'surprise':-1/2,
        'angry':1/10,
        'mustache':-1,
        'old':1/12,
        'gender':1/5   
    }
    
    MODEL_DIR='./checkpoints'  
    attribute_prefix='./Regression/pretrain_weight_final'