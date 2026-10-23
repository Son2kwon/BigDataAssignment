#!/usr/bin/env bash
# 프로젝트 데이터 다운로드 스크립트 (macOS / Linux / Windows Git Bash·WSL)
#
#   bash data/download.sh                      # 기본 데이터셋 3개 (swe-agent, openhands, terminalbench)
#   bash data/download.sh swe-agent openhands  # 이름을 주면 그 데이터셋만 받는다
#   bash data/download.sh all                  # swebench까지 전부
#
# 데이터셋
#   swe-agent      nebius/SWE-agent-trajectories              1.1GB  학습·검증·테스트 (주 데이터)
#   openhands      nebius/SWE-rebench-openhands-trajectories  2.1GB  다른 프레임워크·모델 (OpenHands, Qwen3-Coder)
#   terminalbench  yoonholee/terminalbench-trajectories       0.2GB  도메인 밖 테스트 (Terminal-Bench 2.0)
#   swebench       SWE-bench/experiments 저장소 (git)          0.2GB  외부 테스트용 리더보드 기록. 기본값에서 빠짐
#
# 다시 실행해도 안전하다. 이미 받은 파일은 SHA-256을 확인하고 건너뛰고,
# 중간에 끊긴 파일은 이어서 받는다.
#
# 환경 변수
#   HF_TOKEN     Hugging Face 토큰. 없어도 되지만 요청 제한에 걸리면 설정한다.
#   HF_ENDPOINT  Hugging Face 미러 주소 (기본값 https://huggingface.co)

set -euo pipefail

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT="${HF_ENDPOINT%/}"

SWEBENCH_REPO="https://github.com/SWE-bench/experiments.git"

DEFAULT_KEYS="swe-agent openhands terminalbench"

# 데이터셋 버전(revision)을 고정해서 팀원 모두 같은 파일을 받게 한다.
# 이름 HF저장소 revision 저장폴더(data/ 아래)
DATASETS="
swe-agent nebius/SWE-agent-trajectories 68195a1450865274106246d0d0296a1d6807b88e swe-agent-trajectories
openhands nebius/SWE-rebench-openhands-trajectories 35455389ab51bf5e2306bfd436ef72d0f98bf882 swe-rebench-openhands-trajectories
terminalbench yoonholee/terminalbench-trajectories 04e8940f5b6736a7ce8d22224fe2f2af74163ed2 terminalbench-trajectories
"

# 이름 저장소안경로 크기(byte) sha256
FILES="
swe-agent data/train-00000-of-00012.parquet 89182938 5a395e8c7bb8ddc4b8f4d268506b3a0e2cf9b5ec3922600117322fe788067a13
swe-agent data/train-00001-of-00012.parquet 94796688 fca106cce0f09891c2fadc032fb304da9ae5a7c31d2a39eb7ec70a7bdd4a9882
swe-agent data/train-00002-of-00012.parquet 91137587 fc28c2ab014c6c90d72026dda9cf8753e37b4b7b128c05c4232980cdcc99f3f7
swe-agent data/train-00003-of-00012.parquet 96591990 b6a4e13118de1792b383c077bf6023881a9ff5d4c171a75e767ffb5ad7c037c5
swe-agent data/train-00004-of-00012.parquet 92560283 d1a8d8d3bafbfd589d32a42d4b0a321f107d9aab170e3dc2f749b0f252a958c5
swe-agent data/train-00005-of-00012.parquet 99091960 c507212cd721512240a1bb87a8554bb7b429a9c03649c1473eb3d9e4673e6aca
swe-agent data/train-00006-of-00012.parquet 85410328 fcad81ec5704cb2dc8502a70a1a6cd86a7032227d8de7a613d904636ee53337c
swe-agent data/train-00007-of-00012.parquet 94992382 65547ce464ae1bbff550eacfcfec251ee1cb9b439744389a66bb97a8d5aa1cdb
swe-agent data/train-00008-of-00012.parquet 93467561 bfbd9a58fc9494b49ebd73d5f5e6836a72ae557cbc7ee59d2b2e3f91e3d44027
swe-agent data/train-00009-of-00012.parquet 93395224 8b4c45d8811f0fbbbc5fc46157312ddccb010c439d9e18289ae3ca3e5387ff09
swe-agent data/train-00010-of-00012.parquet 88668958 419e02daa099343e73d45d298b64e574ee012398ebbe19e5798526e5f1336b12
swe-agent data/train-00011-of-00012.parquet 95071802 7c3ccb843bab5457e29a82d9a36a3d59b8e1778c1d97324643c5f85a0d3f9492
openhands trajectories.parquet 2079503354 14048dd1fcd22ce094b6e85f8a38f223a9ef1327031aaaad052804870212efa1
terminalbench data/train-00000-of-00002.parquet 114356912 e95bc8dc9f3e1b2f3c6c383d0a060064205d87e93e9ac9541847d1c511effdfa
terminalbench data/train-00001-of-00002.parquet 106636988 00d61614054bedcbadda9974cef30fe1e07d1701e7126edfa210f99840ae99b2
"

usage() {
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
}

log()  { printf '%s\n' "$*"; }
warn() { printf '경고: %s\n' "$*" >&2; }
die()  { printf '오류: %s\n' "$*" >&2; exit 1; }

keys=""
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        all) keys="$keys $DEFAULT_KEYS swebench" ;;
        swe-agent|openhands|terminalbench|swebench) keys="$keys $arg" ;;
        *) usage >&2; die "알 수 없는 데이터셋: $arg" ;;
    esac
done
[ -n "$keys" ] || keys="$DEFAULT_KEYS"

# 어디서 실행하든 data/ 기준으로 동작한다.
DATA_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- 도구 확인 ------------------------------------------------------------

if command -v curl >/dev/null 2>&1; then
    DOWNLOADER=curl
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER=wget
else
    die "curl 또는 wget이 필요합니다."
fi

# OS마다 SHA-256 도구가 다르다: Linux·Git Bash는 sha256sum, macOS는 shasum,
# 둘 다 없으면 openssl, Windows 기본 도구인 certutil 순서로 찾는다.
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    elif command -v certutil >/dev/null 2>&1; then
        certutil -hashfile "$1" SHA256 | sed -n '2p' | tr -d ' \r' | tr 'A-F' 'a-f'
    else
        echo ""
    fi
}

if [ -z "$(sha256_of "$0")" ]; then
    warn "SHA-256 도구가 없어 파일 크기로만 검증합니다."
    HAVE_SHA=0
else
    HAVE_SHA=1
fi

size_of() { wc -c < "$1" | tr -d ' \r'; }

verify() {  # verify <파일> <크기> <sha256>
    [ -f "$1" ] || return 1
    [ "$(size_of "$1")" = "$2" ] || return 1
    [ "$HAVE_SHA" = 0 ] || [ "$(sha256_of "$1")" = "$3" ]
}

fetch() {  # fetch <url> <저장 경로>. 같은 경로에 남은 파일이 있으면 이어 받는다.
    local args
    if [ "$DOWNLOADER" = curl ]; then
        args=(-fL --retry 5 --retry-delay 3 --connect-timeout 30 -C - --progress-bar -o "$2")
        [ -z "${HF_TOKEN:-}" ] || args+=(-H "Authorization: Bearer $HF_TOKEN")
        curl "${args[@]}" "$1" </dev/null
    else
        args=(--tries=5 --waitretry=3 --timeout=30 -c -O "$2")
        [ -z "${HF_TOKEN:-}" ] || args+=(--header="Authorization: Bearer $HF_TOKEN")
        wget "${args[@]}" "$1" </dev/null
    fi
}

# ---- Hugging Face 데이터셋 -------------------------------------------------

download_hf() {  # download_hf <이름> <저장소> <revision> <저장폴더>
    local key="$1" repo="$2" rev="$3" out="$DATA_DIR/$4"
    local failed=0 total i=0 path size sha name dest part url
    total=$(printf '%s\n' "$FILES" | grep -c "^$key ")

    mkdir -p "$out"
    log "[$key] $repo@${rev:0:7} -> ${4}/"

    while read -r k path size sha; do
        [ "$k" = "$key" ] || continue
        i=$((i + 1))
        name="${path##*/}"
        dest="$out/$name"
        part="$dest.part"

        if verify "$dest" "$size" "$sha"; then
            log "  ($i/$total) $name 이미 있음"
            continue
        fi
        rm -f "$dest"

        # 받다가 끊긴 .part 파일은 이어 받는다. 다 받았는데 이름만 못 바꾼 경우와
        # 기대 크기 이상인데 검증에 실패한 경우는 이어 받을 수 없으니 따로 처리한다.
        if [ -f "$part" ] && [ "$(size_of "$part")" -ge "$size" ]; then
            if verify "$part" "$size" "$sha"; then
                mv -f "$part" "$dest"
                log "  ($i/$total) $name 이미 있음"
                continue
            fi
            rm -f "$part"
        fi

        log "  ($i/$total) $name 받는 중"
        url="$HF_ENDPOINT/datasets/$repo/resolve/$rev/$path"
        if ! fetch "$url" "$part"; then
            warn "$name 다운로드 실패. 다시 실행하면 이어서 받습니다."
            failed=$((failed + 1))
            continue
        fi

        if verify "$part" "$size" "$sha"; then
            mv -f "$part" "$dest"
        else
            warn "$name 검증 실패 (크기 또는 SHA-256 불일치). 파일을 지웠으니 다시 실행하세요."
            rm -f "$part"
            failed=$((failed + 1))
        fi
    done <<EOF
$FILES
EOF

    if [ "$failed" != 0 ]; then
        warn "[$key] ${failed}개 파일을 받지 못했습니다."
        return 1
    fi
    log "  완료: ${total}개 파일 검증됨"
}

# ---- SWE-bench/experiments (git) -------------------------------------------

download_swebench() {
    command -v git >/dev/null 2>&1 || { warn "swebench에는 git이 필요합니다."; return 1; }
    local dest="$DATA_DIR/swebench-experiments"
    log "[swebench] SWE-bench/experiments -> swebench-experiments/"
    if [ -d "$dest/.git" ]; then
        git -C "$dest" pull --ff-only --depth 1 || return 1
    else
        rm -rf "$dest"
        git clone --depth 1 "$SWEBENCH_REPO" "$dest" || return 1
    fi
    log "  완료. 제출작별 궤적은 해당 폴더에서 python -m analysis.download_logs evaluation/<split>/<제출작> 으로 받습니다."
}

# ---- 실행 ------------------------------------------------------------------

failed_keys=""
done_keys=" "
for key in $keys; do
    case "$done_keys" in *" $key "*) continue ;; esac
    done_keys="$done_keys$key "

    if [ "$key" = swebench ]; then
        download_swebench || failed_keys="$failed_keys $key"
        continue
    fi
    line=$(printf '%s\n' "$DATASETS" | grep "^$key ")
    # shellcheck disable=SC2086
    download_hf $line || failed_keys="$failed_keys $key"
done

[ -z "$failed_keys" ] || die "실패한 데이터셋:${failed_keys}. 다시 실행하면 남은 파일만 받습니다."
log "모두 완료"
