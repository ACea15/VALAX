#!/usr/bin/env bash
# md2org.sh — Convert all Markdown docs to Org-mode format.
#
# Mirrors the docs/ directory structure into docs-org/.
# Requires: pandoc >= 3.0
#
# Usage:
#   ./scripts/md2org.sh              # convert docs/ only
#   ./scripts/md2org.sh --all        # also convert top-level .md files (README, CHANGELOG, etc.)
#
# The resulting .org files preserve LaTeX math ($...$ and $$...$$),
# tables, code blocks, and internal links.

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS_SRC="$PROJ_ROOT/docs"
DOCS_DST="$PROJ_ROOT/docs-org"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------
# Platform-specific sed in-place
# ------------------------------------------------------------------
sed_inplace() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# ------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------
if ! command -v pandoc &>/dev/null; then
    echo -e "${RED}Error:${NC} pandoc is not installed."
    echo "  brew install pandoc   # macOS"
    echo "  apt install pandoc    # Debian/Ubuntu"
    exit 1
fi

PANDOC_MAJOR=$(pandoc --version | head -1 | grep -oE '[0-9]+' | head -1)
if (( PANDOC_MAJOR < 3 )); then
    echo -e "${YELLOW}Warning:${NC} pandoc version < 3 detected. Math conversion may be imperfect."
fi

# ------------------------------------------------------------------
# Fix internal anchor links in an .org file.
#
# Pandoc strips leading numbers from CUSTOM_IDs
# (e.g., markdown anchor "1-foundational-framework" becomes CUSTOM_ID
# "foundational-framework"), but internal [[#...]] links keep the
# original markdown-style slug. This function:
#   1. Collects all CUSTOM_IDs actually present in the file.
#   2. For every [[#anchor]] link, if the anchor doesn't match any
#      CUSTOM_ID, tries stripping leading number prefixes (like
#      "1-", "11-", "21-", "31-", etc.) to find a match.
#   3. Rewrites the link to use the actual CUSTOM_ID.
# ------------------------------------------------------------------
fix_internal_links() {
    local file="$1"

    # Collect existing CUSTOM_IDs (one per line)
    local id_list
    id_list=$(grep ':CUSTOM_ID: ' "$file" 2>/dev/null | sed 's/.*:CUSTOM_ID: //' || true)

    # Nothing to fix if no CUSTOM_IDs
    [[ -z "$id_list" ]] && return 0

    # Collect all internal link targets [[#something][...]]
    local anchors
    anchors=$(grep -o '\[\[#[^]]*\]' "$file" 2>/dev/null | sed 's/\[\[#//;s/\]//' | sort -u || true)

    [[ -z "$anchors" ]] && return 0

    # Normalize a slug for fuzzy matching:
    #   - strip leading number prefix (e.g., "21-")
    #   - collapse double dashes to single
    #   - remove periods
    normalize() {
        echo "$1" | sed -E 's/^[0-9]+-//; s/--+/-/g; s/\.//g'
    }

    # Build a lookup: normalized_id → original_id
    # (one per line, tab-separated)
    local id_lookup=""
    while IFS= read -r cid; do
        [[ -z "$cid" ]] && continue
        local norm
        norm=$(normalize "$cid")
        id_lookup="${id_lookup}${norm}	${cid}"$'\n'
    done <<< "$id_list"

    # For each anchor, check if it exists in CUSTOM_IDs.
    # If not, try normalized fuzzy matching.
    while IFS= read -r anchor; do
        [[ -z "$anchor" ]] && continue

        # Already valid? (exact match in id_list)
        if echo "$id_list" | grep -qxF "$anchor"; then
            continue
        fi

        # Normalize the anchor and look up the matching CUSTOM_ID
        local norm_anchor
        norm_anchor=$(normalize "$anchor")

        local matched_id
        matched_id=$(echo "$id_lookup" | grep -F "${norm_anchor}	" | head -1 | cut -f2 || true)

        if [[ -n "$matched_id" ]]; then
            sed_inplace "s|\\[\\[#${anchor}\\]|[[#${matched_id}]|g" "$file"
        fi
    done <<< "$anchors"
}

# ------------------------------------------------------------------
# Convert a single .md file to .org
#   $1 = source .md path (absolute)
#   $2 = destination .org path (absolute)
# ------------------------------------------------------------------
convert_one() {
    local src="$1"
    local dst="$2"

    mkdir -p "$(dirname "$dst")"

    pandoc \
        --from=markdown+tex_math_dollars+pipe_tables+fenced_code_blocks+backtick_code_blocks+header_attributes \
        --to=org \
        --wrap=preserve \
        "$src" -o "$dst"

    # Post-processing fixes:

    # 1. Prepend #+SETUPFILE with relative path to setup.org
    local dst_dir
    dst_dir="$(cd "$(dirname "$dst")" && pwd)"
    local rel_setup
    rel_setup=$(python3 -c "import os.path; print(os.path.relpath('$DOCS_DST/setup.org', '$dst_dir'))")

    # Insert setupfile directive at the top of the file
    local tmp="${dst}.tmp"
    printf '#+SETUPFILE: %s\n\n' "$rel_setup" > "$tmp"
    cat "$dst" >> "$tmp"
    mv "$tmp" "$dst"

    # 2. Replace .md references with .org in internal links
    sed_inplace 's/\.md\]/\.org]/g' "$dst"
    sed_inplace 's/\.md#/\.org#/g' "$dst"

    # 3. Fix internal anchor links whose number prefixes were stripped
    #    by pandoc's CUSTOM_ID generation.
    fix_internal_links "$dst"
}

# ------------------------------------------------------------------
# Main: convert docs/ tree
# ------------------------------------------------------------------
echo -e "${GREEN}Converting docs/ → docs-org/${NC}"
echo "  Source:      $DOCS_SRC"
echo "  Destination: $DOCS_DST"
echo ""

count=0
fail=0

while IFS= read -r -d '' mdfile; do
    relpath="${mdfile#$DOCS_SRC/}"
    orgfile="$DOCS_DST/${relpath%.md}.org"

    if convert_one "$mdfile" "$orgfile" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $relpath"
        ((count++))
    else
        echo -e "  ${RED}✗${NC} $relpath"
        ((fail++))
    fi
done < <(find "$DOCS_SRC" -name '*.md' -type f -print0 | sort -z)

# ------------------------------------------------------------------
# Optional: top-level .md files (README, CHANGELOG, CONTRIBUTING)
# ------------------------------------------------------------------
if [[ "${1:-}" == "--all" ]]; then
    echo ""
    echo -e "${GREEN}Converting top-level .md files → docs-org/${NC}"

    for mdfile in "$PROJ_ROOT"/*.md; do
        [[ -f "$mdfile" ]] || continue
        basename="$(basename "$mdfile")"
        orgfile="$DOCS_DST/${basename%.md}.org"

        if convert_one "$mdfile" "$orgfile" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $basename"
            ((count++))
        else
            echo -e "  ${RED}✗${NC} $basename"
            ((fail++))
        fi
    done
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo -e "${GREEN}Done.${NC} Converted $count files."
if (( fail > 0 )); then
    echo -e "${RED}$fail file(s) failed.${NC}"
fi
echo ""
echo "To read in Emacs:"
echo "  emacs $DOCS_DST/theory.org"
echo ""
echo "To export to LaTeX/PDF from Emacs:"
echo "  C-c C-e l p   (org-latex-export-to-pdf)"
