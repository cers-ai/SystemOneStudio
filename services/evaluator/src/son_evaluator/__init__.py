"""Evaluation: performance block first, effect block second.

Plan: M3 `[GPU]`. The two blocks are stored separately and never merged,
because the priority ordering (鎬ц兘绗竴) is an architectural input, not a
reporting preference. Blocked on 鎶€鏈柟妗?md Q1 and Q3.
"""
