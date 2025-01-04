from . import BaseController
import numpy as np
from typing import Dict, Optional, Any

class Controller(BaseController):
    def __init__(self):
        self.p_g = 0.5
        self.i_g = 0.05
        self.d_g = -0.1  
        
        self.err_sum = 0
        self.prev_err = 0
        self.err_hist = []
        self.window = 5  
        
        self.base_spd = 20.0
    
    def get_gains(self, v):
        """Adaptive gains based on speed and acceleration"""
        scale = self.base_spd / max(v, 5.0)
        accel_scale = min(1.0, abs(self.prev_err) / 2.0)
        
        return {
            'p': self.p_g * scale * (1.2 - accel_scale),  # Reduce P gain at high accelerations
            'i': self.i_g * scale * (0.8 + accel_scale),  # Increase I gain at high accelerations
            'd': self.d_g * np.sqrt(scale) * (1.0 + accel_scale)  # Increase D gain at high accelerations
        }
    
    def smooth_err(self, e):
        self.err_hist.append(e)
        if len(self.err_hist) > self.window:
            self.err_hist.pop(0)
        return np.mean(self.err_hist)
    
    def bank_adj(self, roll_a):
        return -0.1 * roll_a 
    
    def future_ctrl(self, plan, curr_v):
        steps = min(5, len(plan.lataccel))        
        wts = np.linspace(4.0, 1.0, steps)
        weighted_future = np.array(plan.lataccel[:steps]) * wts
        
        total_wt = np.sum(wts)
        if total_wt == 0:
            return 0.0
        return 0.2 * np.sum(weighted_future) / total_wt
    
    def update(self, target_lataccel: float, current_lataccel: float, 
               state: Any, future_plan: Optional[Any] = None) -> float:        
        #print(f"shape of target_lataccel: {target_lataccel.shape if hasattr(target_lataccel, 'shape') else type(target_lataccel)}")
        #print(f"shape of current_lataccel: {current_lataccel.shape if hasattr(current_lataccel, 'shape') else type(current_lataccel)}")
        #print(f"shape of state: {state.shape if hasattr(state, 'shape') else type(state)}")
        #print(f"shape of future_plan: {future_plan.shape if hasattr(future_plan, 'shape') else type(future_plan)}")        
        e = target_lataccel - current_lataccel
        max_change = 2.0 
        if hasattr(self, 'prev_err'):
            e = np.clip(e, self.prev_err - max_change * 0.1,  self.prev_err + max_change * 0.1)
        smoothed_e = self.smooth_err(e)        
        self.err_sum = np.clip(self.err_sum + smoothed_e, -10, 10)
        err_deriv = smoothed_e - self.prev_err
        self.prev_err = smoothed_e
        gains = self.get_gains(state.v_ego)
        
        p_term = gains['p'] * smoothed_e
        i_term = gains['i'] * self.err_sum
        d_term = gains['d'] * err_deriv        
        ctrl_out = p_term + i_term + d_term + self.bank_adj(state.roll_lataccel) + self.future_ctrl(future_plan, state.v_ego)
        return np.clip(ctrl_out, -2, 2) 