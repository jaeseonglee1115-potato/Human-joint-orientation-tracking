import serial
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
except RuntimeError:
    print("한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False

# --- Helper Functions ---
def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm < 1e-9: return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm

def quaternion_to_euler_angles_rad(q_wxyz):
    w, x, y, z = q_wxyz
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1: pitch = math.copysign(math.pi / 2, sinp)
    else: pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

G_VALUE = 1.0

class EKF_AHRS_Python:
    # EKF 클래스 코드는 이전과 동일
    def __init__(self, initial_q_wxyz=np.array([1.0,0.0,0.0,0.0]), initial_gyro_bias_rad_s=np.zeros(3),
                 P0_q_var=0.01, P0_bg_var=0.0001, gyro_noise_var=1e-5, gyro_bias_rw_var=1e-7,
                 accel_noise_var_g2=1e-3, mag_noise_var_uT2=1.0, magnetic_declination_deg=-8.7):
        self.X = np.concatenate((np.array(initial_q_wxyz), np.array(initial_gyro_bias_rad_s))).reshape(-1,1)
        self.P = np.eye(7); self.P[0:4,0:4]*=P0_q_var; self.P[4:7,4:7]*=P0_bg_var
        self._q_gyro_noise_var = gyro_noise_var; self._q_gyro_bias_rw_var = gyro_bias_rw_var
        self.R_accel = np.eye(3)*accel_noise_var_g2
        self.R_mag = np.eye(3)*mag_noise_var_uT2
        self.gravity_world_g = np.array([[0],[0],[-G_VALUE]])
        self.magnetic_declination_rad = math.radians(magnetic_declination_deg)
        inclination_rad = math.radians(55.0) 
        m_N_h = math.cos(inclination_rad); m_Z_v = -math.sin(inclination_rad)
        self.mag_world_ref_norm = np.array([[m_N_h*math.cos(self.magnetic_declination_rad)],
                                           [m_N_h*math.sin(self.magnetic_declination_rad)],[m_Z_v]])
        if np.linalg.norm(self.mag_world_ref_norm)>1e-6: self.mag_world_ref_norm/=np.linalg.norm(self.mag_world_ref_norm)
        self.current_gyro_rad_s_corrected = np.zeros(3)
        self.current_accel_g_norm = G_VALUE

    def _get_rotation_matrix_from_q_wxyz(self, q_wxyz):
        return Rotation.from_quat([q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]).as_matrix()

    def predict(self, gyro_scaled_rad_s, dt):
        if dt<=1e-9: dt=1e-9
        q_prev=self.X[0:4].flatten(); b_g_prev=self.X[4:7].flatten()
        self.current_gyro_rad_s_corrected = gyro_scaled_rad_s - b_g_prev
        omega_c = self.current_gyro_rad_s_corrected
        omega_mat=np.array([[0,-omega_c[0],-omega_c[1],-omega_c[2]],[omega_c[0],0,omega_c[2],-omega_c[1]],
                             [omega_c[1],-omega_c[2],0,omega_c[0]],[omega_c[2],omega_c[1],-omega_c[0],0]])
        q_pred=q_prev+0.5*(omega_mat@q_prev)*dt; q_pred=normalize_quaternion(q_pred)
        self.X[0:4]=q_pred.reshape(4,1); self.X[4:7]=b_g_prev.reshape(3,1)
        F=np.eye(7); F[0:4,0:4]=np.eye(4)+0.5*omega_mat*dt
        qw,qx,qy,qz=q_prev[0],q_prev[1],q_prev[2],q_prev[3]
        J_q_wc=0.5*np.array([[-qx,-qy,-qz],[qw,-qz,qy],[qz,qw,-qx],[-qy,qx,qw]])
        F[0:4,4:7]=-J_q_wc*dt
        Q_q=np.full(4,self._q_gyro_noise_var*(dt**2)); Q_bg=np.full(3,self._q_gyro_bias_rw_var*dt)
        self.P = F@self.P@F.T + np.diag(np.concatenate((Q_q,Q_bg)))

    def update_accelerometer(self, accel_g_val):
        norm_a=np.linalg.norm(accel_g_val)
        self.current_accel_g_norm = norm_a
        if abs(norm_a-G_VALUE)>0.7*G_VALUE or norm_a<0.3*G_VALUE: return
        z_a=(accel_g_val/norm_a).reshape(3,1); q_pred=self.X[0:4].flatten()
        R_bw=self._get_rotation_matrix_from_q_wxyz(q_pred); g_b_exp=R_bw.T@(-self.gravity_world_g)
        y=z_a-g_b_exp;qw,qx,qy,qz=q_pred[0],q_pred[1],q_pred[2],q_pred[3];gzw=G_VALUE
        H_aq=np.array([[-2*qy*gzw,  2*qz*gzw, -2*qw*gzw,  2*qx*gzw],
                       [ 2*qx*gzw,  2*qw*gzw,  2*qz*gzw,  2*qy*gzw],
                       [ 2*qw*gzw, -2*qx*gzw, -2*qy*gzw,  2*qz*gzw]])
        H_a=np.zeros((3,7)); H_a[:,0:4]=H_aq; S=H_a@self.P@H_a.T+self.R_accel
        try: K=self.P@H_a.T@np.linalg.inv(S)
        except np.linalg.LinAlgError: return
        self.X+=K@y; self.P=(np.eye(7)-K@H_a)@self.P
        self.X[0:4]=normalize_quaternion(self.X[0:4].flatten()).reshape(4,1)

    def update_magnetometer(self, mag_uT_val):
        gyro_norm_dps = np.linalg.norm(self.current_gyro_rad_s_corrected) * (180.0/math.pi)
        accel_norm_g = self.current_accel_g_norm
        if gyro_norm_dps > 20.0: return
        if not (0.85 * G_VALUE < accel_norm_g < 1.15 * G_VALUE): return
        norm_m=np.linalg.norm(mag_uT_val)
        if norm_m<5.0 or norm_m>100.0: return
        z_m=(mag_uT_val/norm_m).reshape(3,1);q_pred=self.X[0:4].flatten()
        R_bw=self._get_rotation_matrix_from_q_wxyz(q_pred);m_b_exp=R_bw.T@self.mag_world_ref_norm
        y=z_m-m_b_exp;mwx,mwy,mwz=self.mag_world_ref_norm[0,0],self.mag_world_ref_norm[1,0],self.mag_world_ref_norm[2,0]
        qw,qx,qy,qz=q_pred[0],q_pred[1],q_pred[2],q_pred[3];H_mq=np.zeros((3,4))
        H_mq[0,0]=2*(qw*mwx+qz*mwy-qy*mwz);H_mq[0,1]=2*(qx*mwx+qy*mwy+qz*mwz);H_mq[0,2]=2*(-qy*mwx+qx*mwy+qw*mwz);H_mq[0,3]=2*(-qz*mwx-qw*mwy+qx*mwz)
        H_mq[1,0]=2*(-qz*mwx+qw*mwy+qx*mwz);H_mq[1,1]=2*(qy*mwx-qw*mwy-qz*mwz);H_mq[1,2]=2*(qx*mwx+qy*mwy+qw*mwz);H_mq[1,3]=2*(-qw*mwx+qy*mwy-qx*mwz)
        H_mq[2,0]=2*(qy*mwx-qx*mwy+qw*mwz);H_mq[2,1]=2*(qz*mwx+qw*mwy-qx*mwz);H_mq[2,2]=2*(qw*mwx+qz*mwy-qy*mwz);H_mq[2,3]=2*(qx*mwx-qy*mwy-qz*mwz)
        H_m=np.zeros((3,7));H_m[:,0:4]=H_mq;S=H_m@self.P@H_m.T+self.R_mag
        try: K=self.P@H_m.T@np.linalg.inv(S)
        except np.linalg.LinAlgError: return
        self.X+=K@y;self.P=(np.eye(7)-K@H_m)@self.P
        self.X[0:4]=normalize_quaternion(self.X[0:4].flatten()).reshape(4,1)

    def get_orientation_quaternion_wxyz(self): return self.X[0:4].flatten()
    def get_gyro_bias_rad_s(self): return self.X[4:7].flatten()

def estimate_initial_orientation_from_live_data(avg_accel_g_calib_val, avg_mag_uT_calib_val, magnetic_declination_rad_val):
    if np.linalg.norm(avg_accel_g_calib_val) < 0.5:
        print(f"Warn: Low accelerometer norm for initial orientation (calibrated): {np.linalg.norm(avg_accel_g_calib_val):.2f} g")
        return np.array([1.0, 0.0, 0.0, 0.0])
    if np.linalg.norm(avg_mag_uT_calib_val) < 5.0:
        print(f"Warn: Low magnetometer norm for initial orientation (calibrated): {np.linalg.norm(avg_mag_uT_calib_val):.2f} uT")
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax, ay, az = avg_accel_g_calib_val
    roll = math.atan2(ay, az) 
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    mx, my, mz = avg_mag_uT_calib_val
    r_rp = Rotation.from_euler('xyz', [roll, pitch, 0], degrees=False)
    mag_rotated_for_yaw = r_rp.apply(np.array([mx, my, mz]))
    yaw = math.atan2(-mag_rotated_for_yaw[1], mag_rotated_for_yaw[0]) + magnetic_declination_rad_val
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    final_r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    q_scipy = final_r.as_quat(); return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

# --- 상수 정의 ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600 
ACCEL_FS_G_CONFIG = 2.0; GYRO_FS_DPS_CONFIG = 250.0; MAG_FS_UT_RAW_APPROX = 4912.0
accel_raw_to_g_scale = ACCEL_FS_G_CONFIG / 32767.0
gyro_raw_to_dps_scale = GYRO_FS_DPS_CONFIG / 32767.0
mag_raw_to_uT_scale = MAG_FS_UT_RAW_APPROX / 32767.0
IDEAL_RAW_VALUE_FOR_1G = 32767.0 / ACCEL_FS_G_CONFIG

RAW_ACCEL_BIAS = np.array([0.0, 0.0, 0.0]) 
RAW_GYRO_BIAS_FOR_INIT = np.array([0.0,0.0,0.0])
RAW_MAG_BIAS = np.array([0.0, 0.0, 0.0])

P0_Q_VAR_CONST = (0.1)**2; P0_BG_VAR_CONST = (np.radians(0.5))**2
GYRO_NOISE_VAR_CONST = (np.radians(0.3))**2 # 이전 제안대로 약간 늘린 값
ACCEL_NOISE_VAR_G2_CONST = (0.05)**2 # 이전 제안대로 약간 늘린 값
MAG_NOISE_VAR_UT2_CONST = (50.0)**2  # 이전 제안대로 약간 늘린 값
EKF_DECLINATION_DEG_CONST = -8.7

# --- 전역 변수 ---
ekf_filter = None
script_start_time = time.time()
stabilization_period_seconds = 5.0 
stabilization_over = False
initial_orientation_set = False

sum_raw_accel_stab_list = []
sum_raw_gyro_stab_list = []
sum_raw_mag_stab_list = []
stabilization_sample_count = 0

# 각가속도 계산용 변수
previous_gyro_rad_s_corrected_for_accel = np.zeros(3)
last_time_for_angular_accel = 0.0

ser = None
last_time_main_loop = 0.0 

dt_animation = 0.02 # 목표 애니메이션 프레임 간격

# 팔 끝 경로 추적용
ARM_LENGTH = 0.28
ARM_VECTOR_BODY = np.array([0, ARM_LENGTH, 0]) 
arm_tip_path_history = [np.copy(ARM_VECTOR_BODY)] 
previous_arm_tip_world = np.copy(ARM_VECTOR_BODY) # 팔 끝 속도 계산용 이전 위치
last_time_for_linear_velocity = 0.0            # 팔 끝 속도 계산용 이전 시간



# --- Matplotlib 시각화 설정 ---
fig = plt.figure(figsize=(10, 10)) 
ax_3d = fig.add_subplot(111, projection='3d')
ax_3d.set_xlabel('X World (m)'); ax_3d.set_ylabel('Y World (m)'); ax_3d.set_zlabel('Z World (m)')
ax_3d.set_title('실시간 센서 자세, 팔 끝 경로, 각가속도 및 팔 끝 속도'); # 제목 변경

sensor_axis_length = 0.2
sensor_x_axis_line, = ax_3d.plot([0, sensor_axis_length], [0, 0], [0, 0], 'r-', label='Sensor X')
sensor_y_axis_line, = ax_3d.plot([0, 0], [0, sensor_axis_length], [0, 0], 'g-', label='Sensor Y')
sensor_z_axis_line, = ax_3d.plot([0, 0], [0, 0], [0, sensor_axis_length], 'b-', label='Sensor Z')
arm_tip_path_line, = ax_3d.plot([ARM_VECTOR_BODY[0]], [ARM_VECTOR_BODY[1]], [ARM_VECTOR_BODY[2]], 'm-', label='Arm Tip Path', alpha=0.7)
current_arm_line, = ax_3d.plot([0, ARM_VECTOR_BODY[0]], [0, ARM_VECTOR_BODY[1]], [0, ARM_VECTOR_BODY[2]], 'k-', alpha=0.5, label='Arm Segment')

plot_limit = ARM_LENGTH + sensor_axis_length + 0.1
ax_3d.set_xlim([-plot_limit, plot_limit]); ax_3d.set_ylim([-plot_limit, plot_limit]); ax_3d.set_zlim([-plot_limit, plot_limit])
ax_3d.legend(loc='upper right'); ax_3d.view_init(elev=30., azim=-60)


try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1); ser.flushInput()
    print(f"연결 성공: {SERIAL_PORT} @ {BAUD_RATE} baud.")
except serial.SerialException as e: print(f"시리얼 포트 연결 오류: {e}"); exit()

def update_plot_orientation_and_arm(frame, 
                                    arg_arm_tip_path_line, 
                                    arg_current_arm_line,
                                    arg_sensor_x_line, 
                                    arg_sensor_y_line, 
                                    arg_sensor_z_line, 
                                    arg_ax_3d_obj):
    global last_time_main_loop, stabilization_over, initial_orientation_set, ekf_filter
    global RAW_ACCEL_BIAS, RAW_GYRO_BIAS_FOR_INIT, RAW_MAG_BIAS
    global sum_raw_accel_stab_list, sum_raw_gyro_stab_list, sum_raw_mag_stab_list, stabilization_sample_count
    global previous_gyro_rad_s_corrected_for_accel, last_time_for_angular_accel
    global script_start_time, arm_tip_path_history
    global previous_arm_tip_world, last_time_for_linear_velocity # 팔 끝 속도 계산용 전역변수 추가

    sensor_origin_safe = np.array([0,0,0])
    default_x_axis_world = sensor_origin_safe + np.array([sensor_axis_length, 0, 0])
    default_y_axis_world = sensor_origin_safe + np.array([0, sensor_axis_length, 0])
    default_z_axis_world = sensor_origin_safe + np.array([0, 0, sensor_axis_length])
    default_arm_tip = sensor_origin_safe + ARM_VECTOR_BODY

    current_time_main_loop = time.time()
    actual_dt_val = current_time_main_loop - last_time_main_loop
    if actual_dt_val < 1e-6 and stabilization_over :
        arg_sensor_x_line.set_data_3d(*zip(sensor_origin_safe, default_x_axis_world)); arg_sensor_y_line.set_data_3d(*zip(sensor_origin_safe, default_y_axis_world)); arg_sensor_z_line.set_data_3d(*zip(sensor_origin_safe, default_z_axis_world)); arg_current_arm_line.set_data_3d(*zip(sensor_origin_safe, default_arm_tip))
        return arg_arm_tip_path_line, arg_current_arm_line, arg_sensor_x_line, arg_sensor_y_line, arg_sensor_z_line
    last_time_main_loop = current_time_main_loop

    r_orientation = None # 루프 시작 시 r_orientation 초기화
    arm_tip_world = None # 현재 프레임의 팔 끝 위치 저장용

    if ser and ser.is_open and ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').rstrip()
            data_str = line.split(',')

            if len(data_str) == 9:
                try:
                    parsed_values = [int(val) for val in data_str]
                    raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz, raw_mx, raw_my, raw_mz = parsed_values
                except ValueError:
                    arg_sensor_x_line.set_data_3d(*zip(sensor_origin_safe, default_x_axis_world)); arg_sensor_y_line.set_data_3d(*zip(sensor_origin_safe, default_y_axis_world)); arg_sensor_z_line.set_data_3d(*zip(sensor_origin_safe, default_z_axis_world)); arg_current_arm_line.set_data_3d(*zip(sensor_origin_safe, default_arm_tip))
                    return arg_arm_tip_path_line, arg_current_arm_line, arg_sensor_x_line, arg_sensor_y_line, arg_sensor_z_line

                current_raw_accel_np = np.array([raw_ax, raw_ay, raw_az]); current_raw_gyro_np = np.array([raw_gx, raw_gy, raw_gz]); current_raw_mag_np = np.array([raw_mx, raw_my, raw_mz])

                if not stabilization_over:
                    sum_raw_accel_stab_list.append(current_raw_accel_np); sum_raw_gyro_stab_list.append(current_raw_gyro_np); sum_raw_mag_stab_list.append(current_raw_mag_np)
                    stabilization_sample_count += 1
                    if (current_time_main_loop - script_start_time) > stabilization_period_seconds:
                        if stabilization_sample_count > 50:
                            avg_raw_accel = np.mean(np.array(sum_raw_accel_stab_list), axis=0); avg_raw_gyro = np.mean(np.array(sum_raw_gyro_stab_list), axis=0); avg_raw_mag = np.mean(np.array(sum_raw_mag_stab_list), axis=0)
                            RAW_ACCEL_BIAS = avg_raw_accel.copy(); RAW_ACCEL_BIAS[2] -= IDEAL_RAW_VALUE_FOR_1G
                            RAW_GYRO_BIAS_FOR_INIT = avg_raw_gyro.copy(); RAW_MAG_BIAS = avg_raw_mag.copy()
                            print("--- 파이썬 RAW 센서 바이어스 교정 완료 ---"); print(f"RAW_ACCEL_BIAS: {RAW_ACCEL_BIAS}"); print(f"RAW_GYRO_BIAS_FOR_INIT: {RAW_GYRO_BIAS_FOR_INIT}"); print(f"RAW_MAG_BIAS: {RAW_MAG_BIAS}")
                            initial_gyro_bias_dps_val = RAW_GYRO_BIAS_FOR_INIT * gyro_raw_to_dps_scale; initial_gyro_bias_rad_s_val = np.radians(initial_gyro_bias_dps_val)
                            ekf_filter = EKF_AHRS_Python(initial_gyro_bias_rad_s=initial_gyro_bias_rad_s_val, magnetic_declination_deg=EKF_DECLINATION_DEG_CONST)
                            accel_g_for_init_calib = (avg_raw_accel - RAW_ACCEL_BIAS) * accel_raw_to_g_scale; mag_uT_for_init_calib = (avg_raw_mag - RAW_MAG_BIAS) * mag_raw_to_uT_scale
                            q_init = estimate_initial_orientation_from_live_data(accel_g_for_init_calib, mag_uT_for_init_calib, ekf_filter.magnetic_declination_rad)
                            ekf_filter.X[0:4] = q_init.reshape(4,1); ekf_filter.X[0:4] = normalize_quaternion(ekf_filter.X[0:4].flatten()).reshape(4,1)
                            initial_orientation_set = True; print(f"EKF 초기화 완료. 초기 Q_wxyz={ekf_filter.X[0:4].flatten()}")
                            
                            previous_gyro_rad_s_corrected_for_accel = ekf_filter.current_gyro_rad_s_corrected.copy()
                            last_time_for_angular_accel = current_time_main_loop
                            
                            q_init_for_arm = ekf_filter.get_orientation_quaternion_wxyz()
                            r_orientation_for_arm_init = None
                            if not (np.any(np.isnan(q_init_for_arm)) or np.any(np.isinf(q_init_for_arm))):
                                try: r_orientation_for_arm_init = Rotation.from_quat([q_init_for_arm[1], q_init_for_arm[2], q_init_for_arm[3], q_init_for_arm[0]])
                                except: r_orientation_for_arm_init = None
                            if r_orientation_for_arm_init: initial_arm_tip_position = r_orientation_for_arm_init.apply(ARM_VECTOR_BODY)
                            else: initial_arm_tip_position = np.copy(ARM_VECTOR_BODY)
                            arm_tip_path_history = [initial_arm_tip_position.copy()]
                            previous_arm_tip_world = initial_arm_tip_position.copy() 
                            last_time_for_linear_velocity = current_time_main_loop

                            print("--- 안정화 기간 종료. 자세, 각가속도, 팔 끝 경로 추적 시작. ---")
                        else: print(f"안정화 샘플 부족 ({stabilization_sample_count}개)."); exit()
                        stabilization_over = True
                    arg_sensor_x_line.set_data_3d(*zip(sensor_origin_safe, default_x_axis_world)); arg_sensor_y_line.set_data_3d(*zip(sensor_origin_safe, default_y_axis_world)); arg_sensor_z_line.set_data_3d(*zip(sensor_origin_safe, default_z_axis_world)); arg_current_arm_line.set_data_3d(*zip(sensor_origin_safe, default_arm_tip))
                    return arg_arm_tip_path_line, arg_current_arm_line, arg_sensor_x_line, arg_sensor_y_line, arg_sensor_z_line

                if ekf_filter and initial_orientation_set:
                    accel_g_val_for_ekf = (current_raw_accel_np - RAW_ACCEL_BIAS) * accel_raw_to_g_scale
                    gyro_rads_val_for_ekf = (current_raw_gyro_np * gyro_raw_to_dps_scale) * (math.pi/180.0)
                    mag_uT_val_for_ekf = (current_raw_mag_np - RAW_MAG_BIAS) * mag_raw_to_uT_scale

                    ekf_filter.predict(gyro_rads_val_for_ekf, actual_dt_val); ekf_filter.update_accelerometer(accel_g_val_for_ekf); ekf_filter.update_magnetometer(mag_uT_val_for_ekf)
                    current_q_wxyz = ekf_filter.get_orientation_quaternion_wxyz(); current_gyro_rad_s_corrected = ekf_filter.current_gyro_rad_s_corrected
                    
                    if not (np.any(np.isnan(current_q_wxyz)) or np.any(np.isinf(current_q_wxyz))):
                        try: r_orientation = Rotation.from_quat([current_q_wxyz[1], current_q_wxyz[2], current_q_wxyz[3], current_q_wxyz[0]])
                        except Exception: r_orientation = None 
                    
                    sensor_origin = np.array([0,0,0])
                    if r_orientation:
                        x_body_viz = np.array([sensor_axis_length, 0, 0]); y_body_viz = np.array([0, sensor_axis_length, 0]); z_body_viz = np.array([0, 0, sensor_axis_length])
                        x_axis_world_viz = r_orientation.apply(x_body_viz); y_axis_world_viz = r_orientation.apply(y_body_viz); z_axis_world_viz = r_orientation.apply(z_body_viz)
                        arg_sensor_x_line.set_data_3d([sensor_origin[0], x_axis_world_viz[0]], [sensor_origin[1], x_axis_world_viz[1]], [sensor_origin[2], x_axis_world_viz[2]])
                        arg_sensor_y_line.set_data_3d([sensor_origin[0], y_axis_world_viz[0]], [sensor_origin[1], y_axis_world_viz[1]], [sensor_origin[2], y_axis_world_viz[2]])
                        arg_sensor_z_line.set_data_3d([sensor_origin[0], z_axis_world_viz[0]], [sensor_origin[1], z_axis_world_viz[1]], [sensor_origin[2], z_axis_world_viz[2]])
                        
                        arm_tip_world = r_orientation.apply(ARM_VECTOR_BODY) # 현재 팔 끝 위치 계산
                        arm_tip_path_history.append(arm_tip_world.copy()) # 경로에 추가
                        if len(arm_tip_path_history) > 300: arm_tip_path_history.pop(0)
                        path_xyz_arm_tip = np.array(arm_tip_path_history)
                        arg_arm_tip_path_line.set_data_3d(path_xyz_arm_tip[:,0], path_xyz_arm_tip[:,1], path_xyz_arm_tip[:,2])
                        arg_current_arm_line.set_data_3d([sensor_origin[0], arm_tip_world[0]], [sensor_origin[1], arm_tip_world[1]], [sensor_origin[2], arm_tip_world[2]])
                    else: 
                        arg_sensor_x_line.set_data_3d(*zip(sensor_origin_safe, default_x_axis_world)); arg_sensor_y_line.set_data_3d(*zip(sensor_origin_safe, default_y_axis_world)); arg_sensor_z_line.set_data_3d(*zip(sensor_origin_safe, default_z_axis_world)); arg_current_arm_line.set_data_3d(*zip(sensor_origin_safe, default_arm_tip))
                        arm_tip_world = previous_arm_tip_world # r_orientation이 없으면 이전 위치 사용 (속도 계산 시 0이 되도록)

                    # --- 각가속도 계산 ---
                    angular_acceleration_deg_s2_display = np.zeros(3) # 기본값
                    dt_angular_accel = current_time_main_loop - last_time_for_angular_accel
                    if dt_angular_accel > 1e-5 : 
                        angular_acceleration_rad_s2 = (current_gyro_rad_s_corrected - previous_gyro_rad_s_corrected_for_accel) / dt_angular_accel
                        angular_acceleration_deg_s2_display = np.degrees(angular_acceleration_rad_s2)
                    previous_gyro_rad_s_corrected_for_accel = current_gyro_rad_s_corrected.copy()
                    last_time_for_angular_accel = current_time_main_loop
                    
                    # --- 팔 끝 선형 속력 계산 (외적 사용) ---
                    linear_speed_mps = 0.0
                    angular_speed_dps_display = 0.0
                    if r_orientation: # 유효한 자세와 각속도가 있을 때만 계산
                        omega_body_rad_s = current_gyro_rad_s_corrected
                        omega_world_rad_s = r_orientation.apply(omega_body_rad_s)
                        # arm_vector_world는 위에서 arm_tip_world로 이미 계산됨
                        velocity_tip_world_mps_vector = np.cross(omega_world_rad_s, arm_tip_world) # arm_tip_world는 r_orientation.apply(ARM_VECTOR_BODY)
                        linear_speed_mps = np.linalg.norm(velocity_tip_world_mps_vector)
                        
                        angular_speed_rad_s = np.linalg.norm(omega_body_rad_s)
                        angular_speed_dps_display = np.degrees(angular_speed_rad_s)
                    
                    # --- 최종 출력 ---
                    euler_deg_display = np.zeros(3) # 기본값
                    if current_q_wxyz is not None and not (np.any(np.isnan(current_q_wxyz)) or np.any(np.isinf(current_q_wxyz))):
                        euler_rad = quaternion_to_euler_angles_rad(current_q_wxyz)
                        euler_deg_display = np.degrees(euler_rad)
                    if linear_speed_mps > 0.5:
                        print(f"AngAccel(dps^2): X={angular_acceleration_deg_s2_display[0]:6.2f}, Y={angular_acceleration_deg_s2_display[1]:6.2f}, Z={angular_acceleration_deg_s2_display[2]:6.2f} | Roll: {euler_deg_display[0]:6.2f}, Pitch: {euler_deg_display[1]:6.2f}, Yaw: {euler_deg_display[2]:6.2f} | AngSpeed(dps): {angular_speed_dps_display:6.2f} | ArmTipSpeed(m/s): {linear_speed_mps:5.2f}")

            else: pass 
        except UnicodeDecodeError: pass
        except ValueError: pass
        except Exception as e: print(f"update_plot_orientation_and_arm 내 일반 오류: {type(e).__name__}: {e}, Line: {e.__traceback__.tb_lineno if e.__traceback__ else 'N/A'}")
    else: 
        arg_sensor_x_line.set_data_3d(*zip(sensor_origin_safe, default_x_axis_world)); arg_sensor_y_line.set_data_3d(*zip(sensor_origin_safe, default_y_axis_world)); arg_sensor_z_line.set_data_3d(*zip(sensor_origin_safe, default_z_axis_world)); arg_current_arm_line.set_data_3d(*zip(sensor_origin_safe, default_arm_tip))

    return arg_arm_tip_path_line, arg_current_arm_line, arg_sensor_x_line, arg_sensor_y_line, arg_sensor_z_line


if __name__ == "__main__":
    ser = None 
    try:
        print(f"시도: {SERIAL_PORT} @ {BAUD_RATE} baud 연결 중...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flushInput()
        print(f"연결 성공: {SERIAL_PORT} @ {BAUD_RATE} baud.")
    except serial.SerialException as e: print(f"시리얼 포트 연결 오류 (SerialException): {e}"); exit()
    except PermissionError as pe: print(f"시리얼 포트 연결 오류 (PermissionError): {pe}"); exit()
    except Exception as ex: print(f"시리얼 포트 연결 중 알 수 없는 오류: {ex}"); exit()

    last_time_main_loop = time.time(); 
    last_time_for_angular_accel = last_time_main_loop 
    # previous_arm_tip_world와 last_time_for_linear_velocity는 안정화 루프 후 초기화됨

    ani = FuncAnimation(fig, update_plot_orientation_and_arm,
                        fargs=(arm_tip_path_line, current_arm_line, sensor_x_axis_line, sensor_y_axis_line, sensor_z_axis_line, ax_3d),
                        interval=max(1, int(dt_animation * 1000 - 10)), 
                        blit=True, cache_frame_data=False)
    try:
        plt.tight_layout(); plt.show()
    except KeyboardInterrupt: print("사용자에 의해 프로그램 중지됨.")
    finally:
        if ser and ser.is_open: ser.close(); print("시리얼 포트가 닫혔습니다.")
        print("스크립트를 종료합니다.")