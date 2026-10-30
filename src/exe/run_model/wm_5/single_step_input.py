import numpy as np

from src.model.wm_4 import wm_4
from src.model.wm_5 import response_library, wm_5
from src.validation.exe.run_model.wm_5.single_step import single_step_validation


def build(model_name="wm_5", *, duration=24.0, dt=0.02, step_at=2.0, price_before=0.0, price_after=-1.0):
    duration, dt, step_at, price_before, price_after = single_step_validation.settings(
        duration, dt, step_at, price_before, price_after)
    steps = int(round(duration/dt)) + 1
    T = (0.0, duration)
    baseline = lambda _t: 0.5
    price = lambda t: price_before if t < step_at else price_after
    times = lambda t: t
    if model_name == "wm_4":
        model = wm_4(T, steps, baseline)
        coeff = np.array([-0.25, -0.20, -0.80])
        state = model.deterministic_solver([0.0, 0.0], {
            "r": coeff, "U": price, "Time": times})
        contributions = np.column_stack((coeff[0]*state["F"], coeff[2]*state["R"],
                                         coeff[1]*state["U"]))
        responses = (lambda x: x, lambda x: x, lambda x: x)
    elif model_name == "wm_5":
        library = response_library()
        model = wm_5(T, steps, baseline, [library["linear"], library["cubic"]], [library["linear"]])
        coeff = np.array([-0.25, -0.70, -0.10, -0.20])
        state = model.deterministic_solver([0.0, 0.0], {
            "coefficients": coeff, "U": price, "Time": times})
        contributions = np.column_stack((
            coeff[0]*state["F"], coeff[1]*state["R"]+coeff[2]*state["R"]**3, coeff[3]*state["U"]))
        responses = (lambda x: x, lambda x: (0.7*x+0.1*x**3)/0.8, lambda x: x)
    else: raise ValueError("model_name must be wm_4 or wm_5.")
    # The response h(U)=U is increasing, while theta_U<=0 makes higher prices reduce demand.
    theta_U = float(coeff[1] if model_name == "wm_4" else coeff[-1])
    single_step_validation.price_response(theta_U, responses[2])
    return {"model": model, "state": state, "coefficients": coeff,
            "contributions": contributions, "responses": responses,
            "step_at": step_at, "model_name": model_name}
