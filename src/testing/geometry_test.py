from src.vectors import DesignVector, ASBDesignVector

dv = DesignVector()
asb_dv = ASBDesignVector.from_design_vector(dv)
airplane = asb_dv.make_airplane()
airplane.draw_three_view()