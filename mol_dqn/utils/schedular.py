class PiecewiseSchedule:
    def __init__(self, endpoints, outside_value=None):
        """
        Args:
            endpoints: list of (time, value) pairs
            outside_value: value to return when t is outside the range
        """
        self.endpoints = endpoints
        self.outside_value = outside_value
    
    def value(self, t):
        # 修复：添加边界处理
        if not self.endpoints:
            return self.outside_value
        
        # 如果 t 在第一个点之前
        if t < self.endpoints[0][0]:
            if self.outside_value is not None:
                return self.outside_value
            return self.endpoints[0][1]
        
        # 如果 t 在最后一个点之后
        if t > self.endpoints[-1][0]:
            if self.outside_value is not None:
                return self.outside_value
            return self.endpoints[-1][1]
        
        # 在范围内的线性插值
        for (l_t, l), (r_t, r) in zip(self.endpoints[:-1], self.endpoints[1:]):
            if l_t <= t <= r_t:
                if l_t == r_t:  # 避免除零
                    return l
                alpha = float(t - l_t) / (r_t - l_t)
                return l + alpha * (r - l)
        
        # 如果没有匹配到，返回最后一个值
        return self.endpoints[-1][1]


class LinearSchedule:
    def __init__(self, schedule_timesteps, initial_p, final_p):
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p
    
    def value(self, t):
        if self.schedule_timesteps <= 0:  # 修复：避免除零
            return self.final_p
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)