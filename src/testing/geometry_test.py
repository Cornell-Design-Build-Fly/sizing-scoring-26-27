from src.design_vector import ASBDesignVector

dv = ASBDesignVector(
    wing_span=1.181354,
    wing_chord=0.307086,
    tail_arm=0.845058,
    nose_length=0.20,
)
airplane, s_ref, c_ref, b_ref = dv.make_airplane()
airplane.draw_three_view()