import serial
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation

try:
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
    # plt.rcParams['font.family'] = 'AppleGothic' # macOS
except RuntimeError:
    print("한글 폰트를 찾을 수 없습니다. Malgun Gothic (Windows) 또는 AppleGothic (macOS)을 설치하거나 다른 한글 지원 폰트로 변경해주세요.")
    # 예: 나눔고딕 (NanumGothic) - 시스템에 설치 필요
    # try:
    #     plt.rcParams['font.family'] = 'NanumGothic'
    # except RuntimeError:
    #     print("NanumGothic 폰트도 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# --- Helper Functions ---
def quaternion_to_euler(q_wxyz):
    w, x, y, z = q_wxyz
    sinr_cosp = 2 * (w * x + y * z); cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1: pitch = math.copysign(math.pi / 2, sinp)
    else: pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y); cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm < 1e-9: return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm

def quaternion_multiply(q1_wxyz, q2_wxyz):
    w1,x1,y1,z1=q1_wxyz; w2,x2,y2,z2=q2_wxyz
    w=w1*w2-x1*x2-y1*y2-z1*z2; x=w1*x2+x1*w2+y1*z2-z1*y2
    y=w1*y2-x1*z2+y1*w2+z1*x2; z=w1*z2+x1*y2-y1*x2+z1*w2
    return np.array([w,x,y,z])

G_VALUE = 1.0
GRAVITY_MAGNITUDE = 9.807

class EKF_AHRS_Python:
    def __init__(self, initial_q_wxyz=np.array([1.0,0.0,0.0,0.0]), initial_gyro_bias_rad_s=np.zeros(3),
                 P0_q_var=0.01, P0_bg_var=0.0001, gyro_noise_var=1e-5, gyro_bias_rw_var=1e-7,
                 accel_noise_var_g2=1e-3, mag_noise_var_uT2=1.0, magnetic_declination_deg=-8.5):
        self.X = np.concatenate((np.array(initial_q_wxyz), np.array(initial_gyro_bias_rad_s))).reshape(-1,1)
        self.P = np.eye(7); self.P[0:4,0:4]*=P0_q_var; self.P[4:7,4:7]*=P0_bg_var
        self._q_gyro_noise_var = gyro_noise_var; self._q_gyro_bias_rw_var = gyro_bias_rw_var
        self.R_accel = np.eye(3)*accel_noise_var_g2; self.R_mag = np.eye(3)*mag_noise_var_uT2
        self.gravity_world_g = np.array([[0],[0],[-G_VALUE]])
        self.magnetic_declination_rad = math.radians(magnetic_declination_deg)
        inclination_rad = math.radians(54.5)
        m_N_h = math.cos(inclination_rad); m_Z_v = -math.sin(inclination_rad)
        self.mag_world_ref_norm = np.array([[m_N_h*math.cos(self.magnetic_declination_rad)],
                                           [m_N_h*math.sin(self.magnetic_declination_rad)],[m_Z_v]])
        if np.linalg.norm(self.mag_world_ref_norm)>1e-6: self.mag_world_ref_norm/=np.linalg.norm(self.mag_world_ref_norm)
        print(f"EKF_AHRS_Python initialized. Declination: {magnetic_declination_deg:.2f} deg ({self.magnetic_declination_rad:.4f} rad)")

    def _get_rotation_matrix_from_q_wxyz(self, q_wxyz):
        return Rotation.from_quat([q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]).as_matrix()

    def predict(self, gyro_scaled_rad_s, dt):
        if dt<=1e-9: dt=1e-9
        q_prev=self.X[0:4].flatten(); b_g_prev=self.X[4:7].flatten(); omega_c=gyro_scaled_rad_s-b_g_prev
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
        if abs(norm_a-G_VALUE)>0.8*G_VALUE or norm_a<0.2*G_VALUE:
            return
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
        norm_m=np.linalg.norm(mag_uT_val)
        if norm_m<5.0 or norm_m>120.0:
            return
        z_m=(mag_uT_val/norm_m).reshape(3,1);q_pred=self.X[0:4].flatten()
        R_bw=self._get_rotation_matrix_from_q_wxyz(q_pred);m_b_exp=R_bw.T@self.mag_world_ref_norm
        y=z_m-m_b_exp;mwx,mwy,mwz=self.mag_world_ref_norm[0,0],self.mag_world_ref_norm[1,0],self.mag_world_ref_norm[2,0]
        qw,qx,qy,qz=q_pred[0],q_pred[1],q_pred[2],q_pred[3];H_mq=np.zeros((3,4))
        H_mq[0,0]=2*(qw*mwx+qz*mwy-qy*mwz);H_mq[0,1]=2*(qx*mwx+qy*mwy+qz*mwz);H_mq[0,2]=2*(-qy*mwx+qx*mwy+qw*mwz);H_mq[0,3]=2*(-qz*mwx-qw*mwy+qx*mwz)
        H_mq[1,0]=2*(-qz*mwx+qw*mwy+qx*mwz);H_mq[1,1]=2*(qy*mwx-qw*mwy-qz*mwz);H_mq[1,2]=2*(qx*mwx+qy*mwy-qx*mwz);H_mq[1,3]=2*(qw*mwx+qy*mwy+qz*mwz)
        H_mq[2,0]=2*(qy*mwx-qx*mwy+qw*mwz);H_mq[2,1]=2*(qz*mwx+qw*mwy-qx*mwz);H_mq[2,2]=2*(qw*mwx+qz*mwy-qy*mwz);H_mq[2,3]=2*(qx*mwx+qy*mwy+qz*mwz)

        H_m=np.zeros((3,7));H_m[:,0:4]=H_mq;S=H_m@self.P@H_m.T+self.R_mag
        try: K=self.P@H_m.T@np.linalg.inv(S)
        except np.linalg.LinAlgError: return
        self.X+=K@y;self.P=(np.eye(7)-K@H_m)@self.P
        self.X[0:4]=normalize_quaternion(self.X[0:4].flatten()).reshape(4,1)

    def get_orientation_quaternion_wxyz(self): return self.X[0:4].flatten()
    def get_gyro_bias_rad_s(self): return self.X[4:7].flatten()

def estimate_initial_orientation_from_live_data(avg_accel_g_val, avg_mag_uT_val, magnetic_declination_rad_val):
    if np.linalg.norm(avg_accel_g_val)<1e-3 or np.linalg.norm(avg_mag_uT_val)<1e-3:print("Warn:Low accel/mag for init orient.");return np.array([1.0,0.0,0.0,0.0])
    acc_b_up=-avg_accel_g_val/np.linalg.norm(avg_accel_g_val)
    mag_b=avg_mag_uT_val/np.linalg.norm(avg_mag_uT_val)
    body_Y_temp=np.cross(acc_b_up,mag_b)
    if np.linalg.norm(body_Y_temp)<1e-3:print("Warn:Accel/Mag parallel for init orient.");return np.array([1.0,0.0,0.0,0.0])
    body_Y=body_Y_temp/np.linalg.norm(body_Y_temp)
    body_X=np.cross(body_Y,acc_b_up)
    R_mnb=np.column_stack((body_X,body_Y,acc_b_up))
    try:r_gb=Rotation.from_matrix(R_mnb)
    except ValueError:print("Error:R_mnb not valid rotation matrix for init.");return np.array([1.0,0.0,0.0,0.0])
    r_decl=Rotation.from_euler('z',-magnetic_declination_rad_val,degrees=False)
    r_tnb=r_decl*r_gb
    q_ixyzw=r_tnb.as_quat()
    return np.array([q_ixyzw[3],q_ixyzw[0],q_ixyzw[1],q_ixyzw[2]])

SERIAL_PORT = 'COM3'
BAUD_RATE = 38400
ACCEL_FS_G_CONFIG = 2.0
GYRO_FS_DPS_CONFIG = 250.0
MAG_FS_UT_RAW_APPROX = 4912.0

accel_raw_to_g_scale = ACCEL_FS_G_CONFIG / 32767.0
gyro_raw_to_dps_scale = GYRO_FS_DPS_CONFIG / 32767.0
mag_raw_to_uT_scale = MAG_FS_UT_RAW_APPROX / 32767.0
IDEAL_RAW_VALUE_FOR_1G = 32767.0 / ACCEL_FS_G_CONFIG

RAW_ACCEL_BIAS = np.array([0.0, 0.0, 0.0])
RAW_GYRO_BIAS_FOR_INIT = np.array([0.0,0.0,0.0])
RAW_MAG_BIAS = np.array([0.0,0.0,0.0])

P0_Q_VAR_CONST = (0.1)**2
P0_BG_VAR_CONST = (np.radians(0.5))**2
GYRO_NOISE_VAR_CONST = (np.radians(0.1))**2
GYRO_BIAS_RW_VAR_CONST = (np.radians(0.005))**2
ACCEL_NOISE_VAR_G2_CONST = (0.05)**2
MAG_NOISE_VAR_UT2_CONST = (10.0)**2
EKF_DECLINATION_DEG_CONST = -8.5

ekf_filter = None

script_start_time = time.time()
stabilization_period_seconds = 10.0
stabilization_over = False
initial_orientation_set = False
sum_raw_accel=np.zeros(3); sum_raw_gyro=np.zeros(3); sum_raw_mag=np.zeros(3)
stabilization_sample_count=0

STATIONARY_GYRO_THRESHOLD_RPS = np.radians(2.5)
STATIONARY_WINDOW_SIZE = 30
LINEAR_ACCEL_WORLD_XY_STATIONARY_THRESHOLD_MPS2 = 0.015
LINEAR_ACCEL_WORLD_Z_STATIONARY_THRESHOLD_MPS2  = 0.05
LINEAR_ACCEL_COMPONENT_THRESHOLD_MPS2 = 0.05 # Correctly defined global variable

gyro_buffer_for_zupt = []
linear_accel_world_buffer = []

zupt_engaged = False
last_zupt_engaged_x_position = 0.0
last_zupt_engaged_y_position = 0.0
last_zupt_engaged_z_position = 0.0

dt_animation = 0.02
position = np.array([0.0,0.0,0.0])
velocity = np.array([0.0,0.0,0.0])
path_history = [position.copy()]
ser = None
last_time = time.time()

fig=plt.figure(figsize=(12,9)); ax_3d=fig.add_subplot(111,projection='3d')
ax_3d.set_xlabel('X (m)'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Z (m)')
ax_3d.set_title('실시간 3D 센서 경로 (EKF AHRS + 개선된 ZUPT)'); path_line, = ax_3d.plot([0],[0],[0],'b-',label='경로',alpha=0.7)
arrow_length=0.15
x_arrow, = ax_3d.plot([],[],[],'r-',lw=3,label='센서 X (바디)'); y_arrow, = ax_3d.plot([],[],[],'g-',lw=3,label='센서 Y (바디)'); z_arrow, = ax_3d.plot([],[],[],'purple',lw=3,label='센서 Z (바디)')
ax_3d.legend(loc='upper left'); ax_3d.set_xlim([-0.5,0.5]); ax_3d.set_ylim([-0.5,0.5]); ax_3d.set_zlim([-0.5,0.5])
ax_3d.view_init(elev=20., azim=-35)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1); ser.flushInput()
    print(f"연결 성공: {SERIAL_PORT} @ {BAUD_RATE} baud.")
except serial.SerialException as e: print(f"시리얼 포트 연결 오류: {e}"); exit()

def update_plot(frame, _path_line, _x_arrow, _y_arrow, _z_arrow, _ax_3d):
    global position, velocity, last_time, path_history, ser
    global script_start_time, stabilization_period_seconds, stabilization_over, zupt_engaged
    global RAW_ACCEL_BIAS, RAW_GYRO_BIAS_FOR_INIT, RAW_MAG_BIAS
    global sum_raw_accel, sum_raw_gyro, sum_raw_mag, stabilization_sample_count
    global gyro_buffer_for_zupt, linear_accel_world_buffer
    global initial_orientation_set, ekf_filter
    global last_zupt_engaged_x_position, last_zupt_engaged_y_position, last_zupt_engaged_z_position
    # LINEAR_ACCEL_COMPONENT_THRESHOLD_MPS2 is already global

    current_time = time.time(); actual_dt_val = current_time - last_time
    if actual_dt_val < 1e-6: return _path_line, _x_arrow, _y_arrow, _z_arrow
    last_time = current_time

    line_data_processed = False

    if ser and ser.is_open and ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').rstrip()
            data_str_raw = line.split(',')
            data_str = [s.strip() for s in data_str_raw if s.strip()]

            if len(data_str) == 9:
                try:
                    parsed_values = [float(val) for val in data_str]
                    raw_ax, raw_ay, raw_az = parsed_values[0], parsed_values[1], parsed_values[2]
                    raw_gx, raw_gy, raw_gz = parsed_values[3], parsed_values[4], parsed_values[5]
                    raw_mx, raw_my, raw_mz = parsed_values[6], parsed_values[7], parsed_values[8]
                except ValueError:
                    return _path_line, _x_arrow, _y_arrow, _z_arrow

                current_raw_accel = np.array([raw_ax, raw_ay, raw_az])
                current_raw_gyro = np.array([raw_gx, raw_gy, raw_gz])
                current_raw_mag = np.array([raw_mx, raw_my, raw_mz])

                if not stabilization_over:
                    sum_raw_accel += current_raw_accel
                    sum_raw_gyro += current_raw_gyro
                    sum_raw_mag += current_raw_mag
                    stabilization_sample_count += 1

                    if (current_time - script_start_time) > stabilization_period_seconds:
                        if stabilization_sample_count > 0:
                            avg_ax_raw = sum_raw_accel[0]/stabilization_sample_count
                            avg_ay_raw = sum_raw_accel[1]/stabilization_sample_count
                            avg_az_raw = sum_raw_accel[2]/stabilization_sample_count
                            RAW_ACCEL_BIAS = np.array([avg_ax_raw, avg_ay_raw, avg_az_raw - IDEAL_RAW_VALUE_FOR_1G])

                            avg_gx_raw = sum_raw_gyro[0]/stabilization_sample_count
                            avg_gy_raw = sum_raw_gyro[1]/stabilization_sample_count
                            avg_gz_raw = sum_raw_gyro[2]/stabilization_sample_count
                            RAW_GYRO_BIAS_FOR_INIT = np.array([avg_gx_raw, avg_gy_raw, avg_gz_raw])

                            avg_mx_raw = sum_raw_mag[0]/stabilization_sample_count
                            avg_my_raw = sum_raw_mag[1]/stabilization_sample_count
                            avg_mz_raw = sum_raw_mag[2]/stabilization_sample_count
                            RAW_MAG_BIAS = np.array([avg_mx_raw, avg_my_raw, avg_mz_raw])

                            print("--- 센서 바이어스 교정 완료 (RAW 값) ---")
                            print(f"RAW_ACCEL_BIAS (X,Y, Z-1G_ideal): {RAW_ACCEL_BIAS}")
                            print(f"RAW_GYRO_BIAS_FOR_INIT: {RAW_GYRO_BIAS_FOR_INIT}")
                            print(f"RAW_MAG_BIAS: {RAW_MAG_BIAS}")

                            initial_gyro_bias_rad_s_val = RAW_GYRO_BIAS_FOR_INIT * gyro_raw_to_dps_scale * (math.pi/180.0)
                            ekf_filter = EKF_AHRS_Python(
                                initial_q_wxyz=np.array([1.0,0.0,0.0,0.0]),
                                initial_gyro_bias_rad_s=initial_gyro_bias_rad_s_val,
                                P0_q_var=P0_Q_VAR_CONST, P0_bg_var=P0_BG_VAR_CONST,
                                gyro_noise_var=GYRO_NOISE_VAR_CONST, gyro_bias_rw_var=GYRO_BIAS_RW_VAR_CONST,
                                accel_noise_var_g2=ACCEL_NOISE_VAR_G2_CONST, mag_noise_var_uT2=MAG_NOISE_VAR_UT2_CONST,
                                magnetic_declination_deg=EKF_DECLINATION_DEG_CONST
                            )

                            accel_g_for_init = (np.array([avg_ax_raw, avg_ay_raw, avg_az_raw]) - RAW_ACCEL_BIAS) * accel_raw_to_g_scale
                            mag_uT_for_init = (np.array([avg_mx_raw, avg_my_raw, avg_mz_raw]) - RAW_MAG_BIAS) * mag_raw_to_uT_scale
                            
                            q_init = estimate_initial_orientation_from_live_data(
                                accel_g_for_init, mag_uT_for_init,
                                ekf_filter.magnetic_declination_rad
                            )
                            ekf_filter.X[0:4] = q_init.reshape(4,1)
                            ekf_filter.X[0:4] = normalize_quaternion(ekf_filter.X[0:4].flatten()).reshape(4,1)
                            initial_orientation_set = True
                            print(f"EKF 초기화 완료. 초기 Q_wxyz={ekf_filter.X[0:4].flatten()}")

                        stabilization_over = True
                        print(f"--- {stabilization_period_seconds}초 안정화 기간 종료. 경로 추적 시작. ---")
                        position = np.array([0.0,0.0,0.0]); velocity = np.array([0.0,0.0,0.0])
                        path_history = [position.copy()]
                        last_time = time.time()
                    return _path_line,_x_arrow,_y_arrow,_z_arrow

                if not (ekf_filter and initial_orientation_set):
                    return _path_line, _x_arrow, _y_arrow, _z_arrow

                accel_raw_after_bias = current_raw_accel - RAW_ACCEL_BIAS
                gyro_raw_for_ekf_input = current_raw_gyro
                mag_raw_after_bias = current_raw_mag - RAW_MAG_BIAS

                accel_g_calib = accel_raw_after_bias * accel_raw_to_g_scale
                gyro_rads_for_ekf_input = gyro_raw_for_ekf_input * gyro_raw_to_dps_scale * (math.pi/180.0)
                mag_uT_calib = mag_raw_after_bias * mag_raw_to_uT_scale

                ekf_filter.predict(gyro_rads_for_ekf_input, actual_dt_val)
                ekf_filter.update_accelerometer(accel_g_calib)
                ekf_filter.update_magnetometer(mag_uT_calib)
                current_q_wxyz = ekf_filter.get_orientation_quaternion_wxyz()

                q_for_scipy = np.array([current_q_wxyz[1],current_q_wxyz[2],current_q_wxyz[3],current_q_wxyz[0]])
                r = Rotation.from_quat(q_for_scipy)

                accel_mps2_body_calib = accel_g_calib * GRAVITY_MAGNITUDE
                accel_world_mps2 = r.apply(accel_mps2_body_calib)
                linear_accel_world_mps2 = accel_world_mps2 - np.array([0,0,GRAVITY_MAGNITUDE])

                gyro_rads_corrected_for_zupt = gyro_rads_for_ekf_input - ekf_filter.get_gyro_bias_rad_s()
                gyro_buffer_for_zupt.append(gyro_rads_corrected_for_zupt.copy())
                linear_accel_world_buffer.append(linear_accel_world_mps2.copy())

                if len(gyro_buffer_for_zupt) > STATIONARY_WINDOW_SIZE:
                    gyro_buffer_for_zupt.pop(0)
                if len(linear_accel_world_buffer) > STATIONARY_WINDOW_SIZE:
                    linear_accel_world_buffer.pop(0)

                currently_meets_stationary_conditions = False
                if len(gyro_buffer_for_zupt) == STATIONARY_WINDOW_SIZE and \
                   len(linear_accel_world_buffer) == STATIONARY_WINDOW_SIZE:

                    avg_gyro_mag_zupt = np.mean([np.linalg.norm(g) for g in gyro_buffer_for_zupt])
                    gyro_stationary = avg_gyro_mag_zupt < STATIONARY_GYRO_THRESHOLD_RPS

                    linear_accel_world_data = np.array(linear_accel_world_buffer)
                    mean_abs_linear_accel_world_x = np.mean(np.abs(linear_accel_world_data[:, 0]))
                    mean_abs_linear_accel_world_y = np.mean(np.abs(linear_accel_world_data[:, 1]))
                    mean_abs_linear_accel_world_z = np.mean(np.abs(linear_accel_world_data[:, 2]))

                    linear_accel_x_stationary = mean_abs_linear_accel_world_x < LINEAR_ACCEL_WORLD_XY_STATIONARY_THRESHOLD_MPS2
                    linear_accel_y_stationary = mean_abs_linear_accel_world_y < LINEAR_ACCEL_WORLD_XY_STATIONARY_THRESHOLD_MPS2
                    linear_accel_z_stationary = mean_abs_linear_accel_world_z < LINEAR_ACCEL_WORLD_Z_STATIONARY_THRESHOLD_MPS2
                    
                    if gyro_stationary and linear_accel_x_stationary and linear_accel_y_stationary and linear_accel_z_stationary:
                        currently_meets_stationary_conditions = True
                
                if currently_meets_stationary_conditions:
                    if not zupt_engaged:
                        print(f"[{time.time()-script_start_time:.2f}초] >>> ZUPT 활성화: 속도 및 위치 고정 <<<")
                        last_zupt_engaged_x_position = position[0]
                        last_zupt_engaged_y_position = position[1]
                        last_zupt_engaged_z_position = position[2]
                    zupt_engaged = True
                    velocity = np.array([0.0,0.0,0.0])
                else: 
                    if zupt_engaged:
                        print(f"[{time.time()-script_start_time:.2f}초] >>> ZUPT 비활성화: 움직임 감지 <<<")
                    zupt_engaged = False
                    
                    # ZUPT 아닐 때, 작은 선형 가속도 성분별로 무시 (노이즈 제거 목적)
                    # ** 여기가 수정된 부분입니다 **
                    if abs(linear_accel_world_mps2[0]) < LINEAR_ACCEL_COMPONENT_THRESHOLD_MPS2:
                        linear_accel_world_mps2[0] = 0.0
                    if abs(linear_accel_world_mps2[1]) < LINEAR_ACCEL_COMPONENT_THRESHOLD_MPS2:
                        linear_accel_world_mps2[1] = 0.0
                    if abs(linear_accel_world_mps2[2]) < LINEAR_ACCEL_COMPONENT_THRESHOLD_MPS2:
                        linear_accel_world_mps2[2] = 0.0
                
                if not zupt_engaged:
                    velocity += linear_accel_world_mps2 * actual_dt_val
                
                position += velocity * actual_dt_val

                if zupt_engaged:
                    position[0] = last_zupt_engaged_x_position
                    position[1] = last_zupt_engaged_y_position
                    position[2] = last_zupt_engaged_z_position

                path_history.append(position.copy())
                if len(path_history)>300: path_history.pop(0)

                path_xyz = np.array(path_history)
                _path_line.set_data_3d(path_xyz[:,0],path_xyz[:,1],path_xyz[:,2])

                origin_plot = position
                x_axis_body_plot = np.array([arrow_length,0,0]); y_axis_body_plot = np.array([0,arrow_length,0]); z_axis_body_plot = np.array([0,0,arrow_length])
                x_axis_world_plot = origin_plot + r.apply(x_axis_body_plot)
                y_axis_world_plot = origin_plot + r.apply(y_axis_body_plot)
                z_axis_world_plot = origin_plot + r.apply(z_axis_body_plot)
                _x_arrow.set_data_3d(*zip(origin_plot,x_axis_world_plot))
                _y_arrow.set_data_3d(*zip(origin_plot,y_axis_world_plot))
                _z_arrow.set_data_3d(*zip(origin_plot,z_axis_world_plot))

                if len(path_xyz)>0:
                    all_plot_points = np.vstack((path_xyz, origin_plot.reshape(1,3),
                                                 x_axis_world_plot.reshape(1,3),
                                                 y_axis_world_plot.reshape(1,3),
                                                 z_axis_world_plot.reshape(1,3)))
                    min_vals_plot=np.min(all_plot_points,axis=0); max_vals_plot=np.max(all_plot_points,axis=0)
                    center_plot=(max_vals_plot+min_vals_plot)/2; span_plot=np.max(max_vals_plot-min_vals_plot)*1.2
                    if span_plot < arrow_length*6: span_plot = arrow_length*6
                    _ax_3d.set_xlim([center_plot[0]-span_plot/2,center_plot[0]+span_plot/2])
                    _ax_3d.set_ylim([center_plot[1]-span_plot/2,center_plot[1]+span_plot/2])
                    _ax_3d.set_zlim([center_plot[2]-span_plot/2,center_plot[2]+span_plot/2])
                line_data_processed = True
            else:
                pass
        except UnicodeDecodeError: print(f"유니코드 디코딩 오류: {line if 'line' in locals() else 'N/A'}"); pass
        except ValueError as e_val: print(f"값 변환 오류: {e_val}. 줄: {line if 'line' in locals() else 'N/A'}"); pass
        except Exception as e: print(f"update_plot 내 일반 오류: {type(e).__name__}: {e}. 줄: {line if 'line' in locals() else 'N/A'}"); pass

    return _path_line,_x_arrow,_y_arrow,_z_arrow

ani=FuncAnimation(fig, update_plot,
                  fargs=(path_line, x_arrow, y_arrow, z_arrow, ax_3d),
                  interval=max(1,int(dt_animation*1000-10)),
                  blit=False,
                  cache_frame_data=False)

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("사용자에 의해 플로팅이 중지되었습니다.")
finally:
    if ser and ser.is_open:
        ser.close()
        print("시리얼 포트가 닫혔습니다.")
    elif ser is None:
        print("시리얼 포트가 열리지 않았거나 열기 전에 오류가 발생했습니다.")
    else:
        print("시리얼 포트가 이미 닫혔거나 열리지 않았습니다.")
    print("스크립트를 종료합니다.")

