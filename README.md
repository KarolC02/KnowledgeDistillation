# 📥 Download Datasets

**Tiny ImageNet:**
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip |
unzip tiny-imagenet-200.zip |

# 🖥️ Basic tmux Commands

| Action | Command |
|-------|---------|
| **Create a new session** | `tmux new -s <name>` |
| **Attach to an existing session** | `tmux attach -t <name>` |
| **List all sessions** | `tmux ls` |
| **Detach from current session** | `Ctrl + b`, then `d` |
| **Split pane vertically** | `Ctrl + b`, then `%` |
| **Split pane horizontally** | `Ctrl + b`, then `"` |
| **Kill the current pane** | `Ctrl + b`, then `x` (confirm with `y`) |

---

**Notes:**
- `Ctrl + b` is the default **prefix** key in tmux.  
- After pressing `Ctrl + b`, release and press the second key (like `%`, `d`, `x`, etc.).  
- To move between panes, use `Ctrl + b`, then arrow keys.