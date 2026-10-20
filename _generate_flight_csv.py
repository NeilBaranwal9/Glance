import csv
import math

def generate_flight():
    with open('mock_flight.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Altitude', 'Velocity', 'Acceleration'])
        
        t = 0.0
        dt = 0.1
        
        # Idle (0 to 5s)
        while t < 5.0:
            writer.writerow([f"{t:.2f}", "0.0", "0.0", "0.0"])
            t += dt
            
        # Boost (5 to 8s)
        t_liftoff = 5.0
        while t < 8.0:
            t_rel = t - t_liftoff
            accel = 50.0  # 5G
            vel = accel * t_rel
            alt = 0.5 * accel * (t_rel ** 2)
            writer.writerow([f"{t:.2f}", f"{alt:.2f}", f"{vel:.2f}", f"{accel:.2f}"])
            t += dt
            
        # Coast to Apogee (8s to ~15s)
        v0 = 50.0 * 3.0  # 150 m/s
        alt0 = 0.5 * 50.0 * 9.0  # 225 m
        t0 = 8.0
        
        while t < 25.0:
            t_rel = t - t0
            accel = -9.81
            vel = v0 + accel * t_rel
            alt = alt0 + v0 * t_rel + 0.5 * accel * (t_rel ** 2)
            
            if alt < 0:
                alt = 0.0
                vel = 0.0
                accel = 0.0
            
            writer.writerow([f"{t:.2f}", f"{alt:.2f}", f"{vel:.2f}", f"{accel:.2f}"])
            t += dt

if __name__ == "__main__":
    generate_flight()
    print("mock_flight.csv generated.")
