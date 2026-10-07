#!/usr/bin/env bash
# sbatch_env.sh FILE [SBATCH_OPTIONS...] -- submit an experiment's SLURM script after filling in
# the ${AC2_*} site settings (account, partition, QOS, log paths, ...) in its #SBATCH header.
# SLURM does not expand environment variables in #SBATCH lines, so this substitutes them from the
# repository's .env (see .env.example) and pipes the result to sbatch, which exports the same
# environment to the job. An #SBATCH line whose variable is set but empty (e.g. AC2_QOS=) is
# dropped; an unset variable is an error.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
f="${1:?usage: sbatch_env.sh FILE [SBATCH_OPTIONS...]}"; shift
# The whole script is rendered first, so an unset variable submits nothing.
_rendered="$(perl -ne '
  my $line = $_; my $empty = 0;
  $line =~ s/\$\{(AC2_[A-Z0-9_]+)\}/defined $ENV{$1} ? (length $ENV{$1} ? $ENV{$1} : do { $empty = 1; "" }) : die "sbatch_env.sh: $1 is not set (see .env.example)\n"/ge;
  next if $empty && $line =~ /^#SBATCH/;
  print $line;
' "$f")"
printf '%s\n' "$_rendered" | sbatch "$@"
