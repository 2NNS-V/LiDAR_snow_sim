import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from matplotlib.colors import ListedColormap

def visualize_bin_file_with_labels(bin_path, method='matplotlib', point_size=1):
    """
    bin 파일을 로드하여 label에 따라 다른 색깔로 시각화
    
    Args:
        bin_path: .bin 파일 경로
        method: 'matplotlib' 또는 'open3d'
        point_size: 포인트 크기
    """
    # bin 파일 로드 (x, y, z, intensity, label 순서)
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    
    # 좌표와 label 분리
    xyz = points[:, :3]  # x, y, z
    intensity = points[:, 3]  # intensity
    labels = points[:, 4].astype(int)  # label
    
    print(f"포인트 수: {len(points)}")
    print(f"좌표 범위: X({xyz[:, 0].min():.2f} ~ {xyz[:, 0].max():.2f}), "
          f"Y({xyz[:, 1].min():.2f} ~ {xyz[:, 1].max():.2f}), "
          f"Z({xyz[:, 2].min():.2f} ~ {xyz[:, 2].max():.2f})")
    print(f"라벨 종류: {np.unique(labels)}")
    
    if method == 'matplotlib':
        visualize_with_matplotlib(xyz, labels, intensity, point_size)
    elif method == 'open3d':
        visualize_with_open3d(xyz, labels, intensity)
    else:
        print("지원하지 않는 방법입니다. 'matplotlib' 또는 'open3d'를 사용하세요.")

def visualize_with_matplotlib(xyz, labels, intensity, point_size=1):
    """matplotlib를 사용한 시각화"""
    fig = plt.figure(figsize=(15, 12))
    
    # 고유한 라벨들
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # 3D 시각화
    ax1 = fig.add_subplot(221, projection='3d')
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], 
                   c=[colors[i]], label=f'Label {label}', s=point_size)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Point Cloud by Labels')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2D Top View (X-Y)
    ax2 = fig.add_subplot(222)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax2.scatter(xyz[mask, 0], xyz[mask, 1], 
                   c=[colors[i]], label=f'Label {label}', s=point_size)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (X-Y)')
    ax2.axis('equal')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2D Side View (X-Z)
    ax3 = fig.add_subplot(223)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax3.scatter(xyz[mask, 0], xyz[mask, 2], 
                   c=[colors[i]], label=f'Label {label}', s=point_size)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (X-Z)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Intensity 분포
    ax4 = fig.add_subplot(224)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax4.hist(intensity[mask], bins=50, alpha=0.7, 
                label=f'Label {label}', color=colors[i])
    
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Count')
    ax4.set_title('Intensity Distribution by Labels')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_with_open3d(xyz, labels, intensity):
    """Open3D를 사용한 고급 시각화"""
    # Point cloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # 라벨에 따른 색상 지정
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # 각 포인트에 색상 할당
    point_colors = np.zeros((len(xyz), 3))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        point_colors[mask] = colors[i][:3]  # RGB만 사용
    
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # 시각화
    print("Open3D 창에서 시각화 중...")
    print("- 마우스로 회전/줌 가능")
    print("- 'Q' 키로 종료")
    
    o3d.visualization.draw_geometries([pcd], 
                                    window_name="LiDAR Point Cloud with Labels",
                                    width=1200, height=800)

def analyze_labels(bin_path):
    """라벨 분포 분석"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    labels = points[:, 4].astype(int)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print("\n=== 라벨 분석 ===")
    for label, count in zip(unique_labels, counts):
        percentage = count / len(labels) * 100
        print(f"Label {label}: {count:,}개 ({percentage:.2f}%)")
    
    # 라벨 분포 시각화
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_labels, counts, color=plt.cm.tab20(np.linspace(0, 1, len(unique_labels))))
    plt.xlabel('Label')
    plt.ylabel('Point Count')
    plt.title('Label Distribution')
    
    # 막대 위에 개수 표시
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                f'{count:,}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 사용 예시
if __name__ == "__main__":
    # bin 파일 경로 설정
    bin_path = '/home/soeun/LiDAR_snow_sim/nuScenes_intensity/001000.bin'
    
    try:
        # 라벨 분석
        analyze_labels(bin_path)
        
        # matplotlib로 시각화
        print("\n=== Matplotlib 시각화 ===")
        visualize_bin_file_with_labels(bin_path, method='matplotlib', point_size=0.5)
        
        # Open3D로 시각화 (더 좋은 3D 경험)
        print("\n=== Open3D 시각화 ===")
        visualize_bin_file_with_labels(bin_path, method='open3d')
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {bin_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

# 특정 라벨만 시각화하고 싶은 경우
def visualize_specific_labels(bin_path, target_labels, method='matplotlib'):
    """특정 라벨들만 시각화"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    
    # 특정 라벨들만 필터링
    mask = np.isin(points[:, 4].astype(int), target_labels)
    filtered_points = points[mask]
    
    if len(filtered_points) == 0:
        print(f"지정한 라벨 {target_labels}에 해당하는 포인트가 없습니다.")
        return
    
    xyz = filtered_points[:, :3]
    labels = filtered_points[:, 4].astype(int)
    intensity = filtered_points[:, 3]
    
    print(f"필터링된 포인트 수: {len(filtered_points)}")
    
    if method == 'matplotlib':
        visualize_with_matplotlib(xyz, labels, intensity)
    elif method == 'open3d':
        visualize_with_open3d(xyz, labels, intensity)