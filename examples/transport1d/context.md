# What the agents need to know about transport1d

A neural operator (the student) is pretrained on problems you generate. It sees an initial field
u0 on a periodic interval [0, 1) and predicts the field after time T under

    u_t + c u_x = ν u_xx

`Problem(velocity=c, diffusivity=ν, time=T, slope, seed)`: `slope` sets how smooth u0 is (larger
is smoother) and `seed` fixes it. The student also sees c·T and ν·T. It will be tested on problems
you are not shown. Generate the problems it learns most from right now: the report tells you
which kinds score well. Limits: velocity [−5, 5], diffusivity [0, 0.1], time [0, 1],
slope [0.5, 6], seed an int in [0, 2³²). Problems whose final field is flat are dropped.
