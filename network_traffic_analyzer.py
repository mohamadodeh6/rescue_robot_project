"""
Real-Time Network Traffic Analyzer
====================================
Uses Scapy to sniff live IP packets and displays real-time statistics
in a Tkinter GUI that refreshes every 2 seconds.

Usage:
    sudo python network_traffic_analyzer.py [--iface INTERFACE]

Requirements:
    pip install scapy
"""

import argparse
import socket
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

import tkinter as tk
from tkinter import ttk

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
except ImportError:
    raise SystemExit("Scapy is required. Install it with: pip install scapy")


# ---------------------------------------------------------------------------
# Stats Aggregation
# ---------------------------------------------------------------------------

@dataclass
class TrafficStats:
    """Holds aggregated network traffic statistics."""

    # total bytes received per source IP
    bytes_per_ip: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # total packets received per source IP
    packets_per_ip: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # packet count per protocol name
    protocol_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # lock to protect concurrent reads/writes from sniffer thread vs UI thread
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, src_ip: str, size: int, protocol: str) -> None:
        """Thread-safe update of all stat buckets."""
        with self._lock:
            self.bytes_per_ip[src_ip] += size
            self.packets_per_ip[src_ip] += 1
            self.protocol_counts[protocol] += 1

    def snapshot(self):
        """Return a consistent copy of the current stats for display."""
        with self._lock:
            return (
                dict(self.bytes_per_ip),
                dict(self.packets_per_ip),
                dict(self.protocol_counts),
            )


# Singleton stats object shared between sniffer thread and UI thread
_stats = TrafficStats()


# ---------------------------------------------------------------------------
# Packet Parsing
# ---------------------------------------------------------------------------

def parse_packet(pkt) -> None:
    """
    Called for every captured packet.
    - Ignores non-IP packets silently.
    - Extracts src IP, dst IP, size, and protocol.
    - Updates global _stats in a thread-safe manner.
    """
    try:
        if not pkt.haslayer(IP):
            return  # ignore non-IP traffic

        ip_layer = pkt[IP]
        src_ip: str = ip_layer.src
        size: int = len(pkt)

        # Determine protocol name
        if pkt.haslayer(TCP):
            protocol = "TCP"
        elif pkt.haslayer(UDP):
            protocol = "UDP"
        elif pkt.haslayer(ICMP):
            protocol = "ICMP"
        else:
            protocol = "OTHER"

        _stats.update(src_ip, size, protocol)

    except (AttributeError, IndexError, TypeError):
        # Silently skip malformed or unexpected packets
        pass


# ---------------------------------------------------------------------------
# Reverse DNS Helper
# ---------------------------------------------------------------------------

# DNS lookup results are resolved in a background thread pool so that
# slow DNS responses never block the UI thread.
_dns_cache: Dict[str, str] = {}
_dns_lock = threading.Lock()
_dns_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dns")


def _do_dns_lookup(ip: str) -> None:
    """
    Perform a blocking reverse DNS lookup and store the result in the cache.
    Intended to be submitted to a background thread pool.
    """
    try:
        hostname = socket.gethostbyaddr(ip)[0]
    except (socket.herror, socket.gaierror, socket.timeout):
        hostname = ip

    with _dns_lock:
        _dns_cache[ip] = hostname


def resolve_hostname(ip: str) -> str:
    """
    Return the cached hostname for *ip*, or the IP itself while the lookup
    is pending.  If no lookup has been queued yet for this IP, one is
    submitted to the background thread pool.
    """
    with _dns_lock:
        if ip in _dns_cache:
            return _dns_cache[ip]
        # Mark as pending so we don't submit duplicate lookups
        _dns_cache[ip] = ip

    # Submit the lookup without blocking the caller
    _dns_executor.submit(_do_dns_lookup, ip)
    return ip


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

class AnalyzerApp(tk.Tk):
    """Main Tkinter application window for the traffic analyzer."""

    REFRESH_INTERVAL_MS = 2000  # refresh every 2 seconds

    def __init__(self, iface: Optional[str] = None):
        super().__init__()

        self.iface = iface
        self.title("Real-Time Network Traffic Analyzer")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        self._build_ui()
        self._start_sniffer()
        self._schedule_refresh()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create all widgets."""
        PAD = {"padx": 10, "pady": 6}
        HEADING_FONT = ("Consolas", 13, "bold")
        LABEL_FONT = ("Consolas", 11)

        # ---- Title bar ------------------------------------------------
        title_lbl = tk.Label(
            self,
            text="🛡  Network Traffic Analyzer",
            font=("Consolas", 16, "bold"),
            fg="#cdd6f4",
            bg="#1e1e2e",
        )
        title_lbl.pack(**PAD)

        # ---- Interface label ------------------------------------------
        iface_text = f"Interface: {self.iface or 'default'}"
        self._iface_lbl = tk.Label(
            self, text=iface_text, font=LABEL_FONT, fg="#a6e3a1", bg="#1e1e2e"
        )
        self._iface_lbl.pack()

        # ---- Top-5 IPs table -----------------------------------------
        frame_top = tk.LabelFrame(
            self,
            text=" Top 5 Source IPs by Bandwidth ",
            font=HEADING_FONT,
            fg="#89b4fa",
            bg="#1e1e2e",
            bd=2,
        )
        frame_top.pack(fill="x", **PAD)

        cols_top = ("Rank", "Source IP", "Hostname", "Bytes", "Packets", "Bandwidth")
        self._tree_top = ttk.Treeview(
            frame_top, columns=cols_top, show="headings", height=5
        )
        col_widths = [50, 130, 200, 100, 90, 120]
        for col, w in zip(cols_top, col_widths):
            self._tree_top.heading(col, text=col)
            self._tree_top.column(col, width=w, anchor="center")
        self._tree_top.pack(fill="x", padx=4, pady=4)

        # ---- Protocol distribution table -----------------------------
        frame_proto = tk.LabelFrame(
            self,
            text=" Protocol Distribution ",
            font=HEADING_FONT,
            fg="#89b4fa",
            bg="#1e1e2e",
            bd=2,
        )
        frame_proto.pack(fill="x", **PAD)

        cols_proto = ("Protocol", "Packets", "Share %")
        self._tree_proto = ttk.Treeview(
            frame_proto, columns=cols_proto, show="headings", height=4
        )
        for col in cols_proto:
            self._tree_proto.heading(col, text=col)
            self._tree_proto.column(col, width=150, anchor="center")
        self._tree_proto.pack(fill="x", padx=4, pady=4)

        # ---- Summary bar ---------------------------------------------
        frame_summary = tk.Frame(self, bg="#313244", bd=1, relief="sunken")
        frame_summary.pack(fill="x", padx=10, pady=(0, 6))

        self._summary_var = tk.StringVar(value="Total Packets: 0   |   Total Bytes: 0 B   |   Unique IPs: 0")
        summary_lbl = tk.Label(
            frame_summary,
            textvariable=self._summary_var,
            font=LABEL_FONT,
            fg="#f38ba8",
            bg="#313244",
            pady=4,
        )
        summary_lbl.pack()

        # ---- Last-updated timestamp ----------------------------------
        self._time_var = tk.StringVar(value="")
        time_lbl = tk.Label(
            self,
            textvariable=self._time_var,
            font=("Consolas", 9),
            fg="#585b70",
            bg="#1e1e2e",
        )
        time_lbl.pack(pady=(0, 4))

        # ---- Apply dark style to Treeview widgets --------------------
        self._apply_treeview_style()

    def _apply_treeview_style(self) -> None:
        """Configure a dark colour scheme for all Treeview widgets."""
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#181825",
            foreground="#cdd6f4",
            fieldbackground="#181825",
            rowheight=22,
            font=("Consolas", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#313244",
            foreground="#89b4fa",
            font=("Consolas", 10, "bold"),
        )
        style.map("Treeview", background=[("selected", "#45475a")])

    # ------------------------------------------------------------------
    # Sniffer Thread
    # ------------------------------------------------------------------

    def _start_sniffer(self) -> None:
        """Launch the Scapy sniffer in a background daemon thread."""
        t = threading.Thread(target=self._sniffer_worker, daemon=True)
        t.start()

    def _sniffer_worker(self) -> None:
        """Blocking Scapy sniff call — runs in its own thread forever."""
        kwargs = {
            "prn": parse_packet,
            "store": False,  # don't accumulate packets in memory
            "filter": "ip",  # BPF filter: capture only IP packets at kernel level
        }
        if self.iface:
            kwargs["iface"] = self.iface
        sniff(**kwargs)

    # ------------------------------------------------------------------
    # Periodic Refresh
    # ------------------------------------------------------------------

    def _schedule_refresh(self) -> None:
        """Ask Tkinter to call _refresh after REFRESH_INTERVAL_MS milliseconds."""
        self._refresh()
        self.after(self.REFRESH_INTERVAL_MS, self._schedule_refresh)

    def _refresh(self) -> None:
        """Pull a stats snapshot and update every widget."""
        bytes_per_ip, packets_per_ip, protocol_counts = _stats.snapshot()

        # ---- Top-5 table ---------------------------------------------
        # Sort IPs by total bytes descending and take the top 5
        sorted_ips = sorted(bytes_per_ip.items(), key=lambda x: x[1], reverse=True)[:5]

        self._tree_top.delete(*self._tree_top.get_children())
        for rank, (ip, total_bytes) in enumerate(sorted_ips, start=1):
            hostname = resolve_hostname(ip)
            packets = packets_per_ip.get(ip, 0)
            bandwidth = _format_bytes(total_bytes)
            self._tree_top.insert(
                "",
                "end",
                values=(rank, ip, hostname, total_bytes, packets, bandwidth),
            )

        # ---- Protocol table ------------------------------------------
        total_proto_pkts = sum(protocol_counts.values()) or 1  # avoid division by zero
        self._tree_proto.delete(*self._tree_proto.get_children())
        for proto, count in sorted(protocol_counts.items(), key=lambda x: x[1], reverse=True):
            share = f"{count / total_proto_pkts * 100:.1f}%"
            self._tree_proto.insert("", "end", values=(proto, count, share))

        # ---- Summary bar ---------------------------------------------
        total_packets = sum(packets_per_ip.values())
        total_bytes = sum(bytes_per_ip.values())
        unique_ips = len(bytes_per_ip)
        self._summary_var.set(
            f"Total Packets: {total_packets:,}   |   "
            f"Total Bytes: {_format_bytes(total_bytes)}   |   "
            f"Unique IPs: {unique_ips}"
        )

        # ---- Timestamp -----------------------------------------------
        self._time_var.set(f"Last updated: {time.strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _format_bytes(num: int) -> str:
    """Convert a byte count to a human-readable string (B / KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Network Traffic Analyzer (Tkinter + Scapy)"
    )
    parser.add_argument(
        "--iface",
        "-i",
        metavar="INTERFACE",
        default=None,
        help="Network interface to sniff on (e.g. eth0, wlan0). "
             "Omit to use Scapy's default interface.",
    )
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = _parse_args()

    app = AnalyzerApp(iface=args.iface)

    # Graceful exit on window close or Ctrl+C
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nNetwork Traffic Analyzer stopped.")


if __name__ == "__main__":
    main()
