#!/bin/bash
# ============================================================
# Oracle Cloud VM 초기 설정 스크립트
# GitHub Actions SSH 연동을 위한 원타임 설정
# ============================================================
# 사용법: VM에 SSH 접속 후 실행
#   chmod +x oracle_setup.sh && ./oracle_setup.sh
# ============================================================

set -e

echo "=========================================="
echo " Oracle VM 초기 설정 시작"
echo "=========================================="

# 1. 시스템 업데이트
echo "[1/6] 시스템 패키지 업데이트..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# 2. Python 3.11 설치
echo "[2/6] Python 3.11 설치..."
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip git

# python3.11을 기본 python3로 설정
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --set python3 /usr/bin/python3.11 2>/dev/null || true

echo "Python 버전: $(python3 --version)"

# 3. 프로젝트 디렉토리 생성 및 레포 클론
echo "[3/6] 프로젝트 설정..."
REPO_DIR="$HOME/upbit"

if [ -d "$REPO_DIR" ]; then
    echo "  기존 디렉토리 발견, git pull..."
    cd "$REPO_DIR"
    git pull --quiet
else
    echo "  레포지토리 URL을 입력하세요 (예: https://github.com/username/upbit.git):"
    read -r REPO_URL
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# 4. Python 가상환경 생성 + 의존성 설치
echo "[4/6] Python 가상환경 및 의존성 설치..."
python3 -m venv "$REPO_DIR/venv"
source "$REPO_DIR/venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$REPO_DIR/requirements.txt" -q
echo "  설치된 패키지: $(pip list --format=columns | wc -l)개"

# 5. SSH 키 확인 (GitHub Actions용)
echo "[5/6] SSH 설정 확인..."
if [ ! -f "$HOME/.ssh/authorized_keys" ]; then
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    touch "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    echo "  authorized_keys 파일 생성됨"
    echo "  ⚠ GitHub Actions SSH 공개키를 여기에 추가해야 합니다!"
else
    echo "  authorized_keys 존재 ($(wc -l < "$HOME/.ssh/authorized_keys")줄)"
fi

# 6. 방화벽 확인
echo "[6/6] 방화벽 설정..."
if command -v ufw &> /dev/null; then
    sudo ufw status | head -5
else
    echo "  ufw 미설치 (Oracle Cloud 보안그룹으로 관리)"
fi

echo ""
echo "=========================================="
echo " 설정 완료!"
echo "=========================================="
echo ""
echo "VM 공개 IP: $(curl -s ifconfig.me 2>/dev/null || echo '확인 불가')"
echo "프로젝트 경로: $REPO_DIR"
echo "Python: $(python3 --version)"
echo ""
echo "다음 단계:"
echo "  1. GitHub Actions SSH 공개키를 ~/.ssh/authorized_keys에 추가"
echo "  2. Upbit API에 위 IP 주소를 등록"
echo "  3. GitHub Secrets에 ORACLE_HOST, ORACLE_USER, ORACLE_SSH_KEY 추가"
echo ""
