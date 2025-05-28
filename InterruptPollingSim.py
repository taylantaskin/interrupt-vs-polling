import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading
import csv
import random
import numpy as np
from matplotlib.figure import Figure
import os

class SimulationEngine:
    """Handles the simulation logic and calculations"""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        # Performance tracking variables
        self.poll_count = 0
        self.interrupt_count = 0
        self.cpu_load = 0
        self.poll_latencies = []
        self.interrupt_latencies = []
        self.throughput_data = []
        self.poll_times = []
        self.interrupt_times = []
        self.running = False
        
        # New tracking variables for CPU and efficiency
        self.cpu_utilization_data = []  # List of (time, usage) tuples

    def start(self, callback):
        """Start the simulation"""
        self.reset()
        self.running = True
        self.callback = callback
        threading.Thread(target=self._run_simulation).start()

    def stop(self):
        """Stop the simulation"""
        self.running = False

    def _run_simulation(self):
        """Run the actual simulation"""
        start_time = time.time()
        last_throughput_calc = start_time
        last_cpu_calc = start_time
        event_window = []  # For calculating rolling throughput

        while self.running and time.time() - start_time < self.config.sim_duration.get():
            current_time = time.time() - start_time

            # Calculate polling interval based on mode
            interval = self.config.polling_interval.get()
            if self.config.mode.get() == 'High Load':
                interval *= random.uniform(0.5, 1.5)
            elif self.config.mode.get() == 'Low Power':
                interval *= random.uniform(1.0, 2.0)

            # Check for polling events
            if int(current_time / interval) != len(self.poll_times):
                self._process_poll_event(current_time)
                event_window.append(time.time())

            # Check for interrupt events
            if int(current_time / self.config.interrupt_interval.get()) != len(self.interrupt_times):
                self._process_interrupt_event(current_time)
                event_window.append(time.time())

            # Calculate CPU load and store utilization data every 0.1 seconds
            if time.time() - last_cpu_calc >= 0.1:
                total_events = max((self.poll_count + self.interrupt_count), 1)
                self.cpu_load = (self.poll_count / total_events) * 100
                
                # Calculate CPU utilization based on recent activity
                recent_events = len([t for t in event_window if time.time() - t <= 0.1])
                cpu_usage = min(recent_events * 10, 100)  # Each event contributes 10% CPU usage
                self.cpu_utilization_data.append((current_time, cpu_usage))
                last_cpu_calc = time.time()

            # Calculate throughput (events per second) every 0.5 seconds
            if time.time() - last_throughput_calc >= 0.5:
                # Remove events older than 1 second from window
                current = time.time()
                event_window = [t for t in event_window if current - t <= 1.0]

                # Calculate events per second in the current window
                throughput = len(event_window)
                self.throughput_data.append((current_time, throughput))
                last_throughput_calc = time.time()

            # Update GUI through callback
            self.callback(current_time)
            time.sleep(0.05)  # Small delay to prevent CPU hogging

        # When simulation completes
        self.running = False
        self.callback(time.time() - start_time, finished=True)

    def _process_poll_event(self, current_time):
        """Process a polling event"""
        # Simulate processing time for polling
        process_start = time.time()
        # Simulate work with variable processing time
        processing_time = random.uniform(0.001, 0.01)  # 1-10ms of processing
        if self.config.mode.get() == 'High Load':
            processing_time *= 2

        # Simulate CPU work
        time.sleep(processing_time)

        self.poll_times.append(current_time)
        self.poll_count += 1
        latency = time.time() - process_start
        self.poll_latencies.append(latency)

    def _process_interrupt_event(self, current_time):
        """Process an interrupt event"""
        # Simulate processing time for interrupt
        process_start = time.time()
        # Simulate work with variable processing time (interrupts usually faster)
        processing_time = random.uniform(0.0005, 0.005)  # 0.5-5ms of processing
        if self.config.mode.get() == 'High Load':
            processing_time *= 2

        # Simulate CPU work
        time.sleep(processing_time)

        self.interrupt_times.append(current_time)
        self.interrupt_count += 1
        latency = time.time() - process_start
        self.interrupt_latencies.append(latency)


class DataExporter:
    """Handles exporting simulation results"""

    def __init__(self, export_dir):
        self.export_dir = export_dir
        # Ensure export directory exists
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def export_csv(self, engine):
        """Export simulation results to CSV"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.export_dir, f'simulation_results_{timestamp}.csv')

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write simulation parameters
            writer.writerow(['===== SIMULATION PARAMETERS ====='])
            writer.writerow(['Polling Interval (s)', engine.config.polling_interval.get()])
            writer.writerow(['Interrupt Interval (s)', engine.config.interrupt_interval.get()])
            writer.writerow(['Simulation Duration (s)', engine.config.sim_duration.get()])
            writer.writerow(['Mode', engine.config.mode.get()])
            writer.writerow([])

            # Summary metrics
            writer.writerow(['===== SUMMARY METRICS ====='])
            writer.writerow(['Total Polling Events', engine.poll_count])
            writer.writerow(['Total Interrupt Events', engine.interrupt_count])
            writer.writerow(['Total Events', engine.poll_count + engine.interrupt_count])
            writer.writerow(['CPU Load (%)', f'{engine.cpu_load:.2f}'])

            # Calculate latency statistics
            poll_latencies_ms = [l * 1000 for l in engine.poll_latencies]
            int_latencies_ms = [l * 1000 for l in engine.interrupt_latencies]
            all_latencies_ms = poll_latencies_ms + int_latencies_ms

            # Write latency metrics
            self._write_latency_metrics(writer, poll_latencies_ms, int_latencies_ms, all_latencies_ms)

            # Write throughput metrics
            self._write_throughput_metrics(writer, engine.throughput_data)

            # Write raw event data
            self._write_raw_events(writer, engine.poll_times, engine.interrupt_times,
                                   engine.poll_latencies, engine.interrupt_latencies)

        return filename

    def _write_latency_metrics(self, writer, poll_latencies_ms, int_latencies_ms, all_latencies_ms):
        """Write latency metrics to CSV"""
        writer.writerow([])
        writer.writerow(['===== LATENCY METRICS (ms) ====='])
        writer.writerow(['Metric', 'Polling', 'Interrupt', 'Combined'])

        # Calculate and write statistics
        metrics = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '95th Percentile']
        functions = [np.mean, np.median, np.std, np.min, np.max, lambda x: np.percentile(x, 95)]

        for metric, func in zip(metrics, functions):
            poll_val = func(poll_latencies_ms) if poll_latencies_ms else 0
            int_val = func(int_latencies_ms) if int_latencies_ms else 0
            all_val = func(all_latencies_ms) if all_latencies_ms else 0
            writer.writerow([metric, f'{poll_val:.3f}', f'{int_val:.3f}', f'{all_val:.3f}'])

    def _write_throughput_metrics(self, writer, throughput_data):
        """Write throughput metrics to CSV"""
        writer.writerow([])
        writer.writerow(['===== THROUGHPUT METRICS (events/s) ====='])

        if throughput_data:
            throughput_values = [tp[1] for tp in throughput_data]
            metrics = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '95th Percentile']
            functions = [np.mean, np.median, np.std, np.min, np.max, lambda x: np.percentile(x, 95)]

            for metric, func in zip(metrics, functions):
                writer.writerow([metric, f'{func(throughput_values):.3f}'])
        else:
            writer.writerow(['No throughput data available'])

    def _write_raw_events(self, writer, poll_times, interrupt_times, poll_latencies, interrupt_latencies):
        """Write raw event data to CSV"""
        writer.writerow([])
        writer.writerow(['===== RAW EVENT DATA ====='])
        writer.writerow(['Time (s)', 'Event Type', 'Latency (ms)'])

        for time_val, latency in zip(poll_times, poll_latencies[:len(poll_times)]):
            writer.writerow([f'{time_val:.3f}', 'Polling', f'{latency * 1000:.3f}'])

        for time_val, latency in zip(interrupt_times, interrupt_latencies[:len(interrupt_times)]):
            writer.writerow([f'{time_val:.3f}', 'Interrupt', f'{latency * 1000:.3f}'])


class ChartManager:
    """Manages simulation charts and visualizations"""

    def __init__(self, parent):
        self.parent = parent
        self._create_charts()
        self.update_interval = 0.5  # Update interval in seconds
        self.last_update_time = 0
        self.data_window_size = 100  # Maximum number of data points to show
        
    def _create_charts(self):
        """Create the chart figures and axes"""
        # Create figure for CPU utilization
        self.cpu_util_fig = Figure(figsize=(6, 2), dpi=100)
        self.cpu_util_ax = self.cpu_util_fig.add_subplot(111)
        self.cpu_util_ax.set_title('CPU Utilization Over Time')
        self.cpu_canvas = FigureCanvasTkAgg(self.cpu_util_fig, master=self.parent)
        self.cpu_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create figure for events plot
        self.events_fig = Figure(figsize=(6, 2), dpi=100)
        self.events_ax = self.events_fig.add_subplot(111)
        self.events_ax.set_title('Interrupt vs Polling Simulation')
        self.events_canvas = FigureCanvasTkAgg(self.events_fig, master=self.parent)
        self.events_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create figure for latency plot
        self.latency_fig = Figure(figsize=(6, 2), dpi=100)
        self.latency_ax = self.latency_fig.add_subplot(111)
        self.latency_ax.set_title('Event Processing Latency')
        self.latency_canvas = FigureCanvasTkAgg(self.latency_fig, master=self.parent)
        self.latency_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create figure for throughput plot
        self.throughput_fig = Figure(figsize=(6, 2), dpi=100)
        self.throughput_ax = self.throughput_fig.add_subplot(111)
        self.throughput_ax.set_title('System Throughput')
        self.throughput_canvas = FigureCanvasTkAgg(self.throughput_fig, master=self.parent)
        self.throughput_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _limit_data_points(self, data_list, max_points=100):
        """Limit the number of data points to improve performance"""
        if len(data_list) > max_points:
            return data_list[-max_points:]
        return data_list

    def update_cpu_utilization(self, time_points, cpu_usage):
        """Update the CPU utilization chart"""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.cpu_util_ax.clear()
        if time_points and cpu_usage:
            # Limit data points
            time_points = self._limit_data_points(time_points)
            cpu_usage = self._limit_data_points(cpu_usage)
            
            self.cpu_util_ax.plot(time_points, cpu_usage, 'r-', label='CPU Usage')
            self.cpu_util_ax.set_xlabel('Time (s)')
            self.cpu_util_ax.set_ylabel('CPU Usage (%)')
            self.cpu_util_ax.set_ylim(0, 100)
            self.cpu_util_ax.grid(True)
            self.cpu_util_ax.legend()
        self.last_update_time = current_time
        self.cpu_canvas.draw_idle()  # Use draw_idle instead of draw

    def update_events_chart(self, poll_times, interrupt_times, cpu_load):
        """Update the events chart"""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.events_ax.clear()
        # Limit data points
        poll_times = self._limit_data_points(poll_times)
        interrupt_times = self._limit_data_points(interrupt_times)
        
        self.events_ax.scatter(poll_times, [1] * len(poll_times), marker='>', color='red', label='Polling', s=100)
        self.events_ax.scatter(interrupt_times, [2] * len(interrupt_times), marker='^', color='blue', label='Interrupt', s=100)
        self.events_ax.set_yticks([1, 2])
        self.events_ax.set_yticklabels(['Polling', 'Interrupt'])
        self.events_ax.set_xlabel('Time (s)')
        self.events_ax.set_title(f'Interrupt vs Polling Simulation - CPU Load: {cpu_load:.2f}%')
        self.events_ax.legend()
        self.events_canvas.draw_idle()

    def update_latency_chart(self, poll_latencies, interrupt_latencies):
        """Update the latency chart"""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.latency_ax.clear()

        # Limit data points
        poll_latencies = self._limit_data_points(poll_latencies)
        interrupt_latencies = self._limit_data_points(interrupt_latencies)

        times = list(range(len(poll_latencies)))
        poll_latencies_ms = [lat * 1000 for lat in poll_latencies]  # Convert to ms

        interrupt_times = list(range(len(interrupt_latencies)))
        interrupt_latencies_ms = [lat * 1000 for lat in interrupt_latencies]  # Convert to ms

        if poll_latencies_ms:
            self.latency_ax.plot(times, poll_latencies_ms, 'r-', label='Polling Latency')
        if interrupt_latencies_ms:
            self.latency_ax.plot(interrupt_times, interrupt_latencies_ms, 'b-', label='Interrupt Latency')

        self.latency_ax.set_xlabel('Event Number')
        self.latency_ax.set_ylabel('Latency (ms)')
        self.latency_ax.set_title('Event Processing Latency')
        self.latency_ax.legend()
        self.latency_canvas.draw_idle()

    def update_throughput_chart(self, throughput_data):
        """Update the throughput chart"""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.throughput_ax.clear()

        if throughput_data:
            # Limit data points
            throughput_data = self._limit_data_points(throughput_data)
            times = [data[0] for data in throughput_data]
            values = [data[1] for data in throughput_data]

            self.throughput_ax.plot(times, values, 'g-')
            self.throughput_ax.set_xlabel('Time (s)')
            self.throughput_ax.set_ylabel('Events per second')
            self.throughput_ax.set_title('System Throughput')
        else:
            self.throughput_ax.set_title('No throughput data yet')

        self.throughput_canvas.draw_idle()


class SimulationConfig:
    """Stores simulation configuration parameters"""

    def __init__(self):
        self.polling_interval = tk.DoubleVar(value=0.1)
        self.interrupt_interval = tk.DoubleVar(value=1.0)
        self.sim_duration = tk.DoubleVar(value=10.0)
        self.mode = tk.StringVar(value='Normal')
        self.export_dir = tk.StringVar(value=os.path.join(os.getcwd(), "simulation_results"))


class InterruptPollingSim:
    """Main application class for the Interrupt vs Polling Simulation"""

    def __init__(self, root):
        self.root = root
        self.root.title('Interrupt vs Polling Simulation')
        self.root.geometry('1200x800')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Set up components
        self.config = SimulationConfig()
        self.create_widgets()
        self.engine = SimulationEngine(self.config)
        self.exporter = DataExporter(self.config.export_dir.get())

    def on_closing(self):
        """Handle window closing event"""
        self.engine.stop()
        self.root.quit()
        self.root.destroy()

    def create_widgets(self):
        """Create the user interface"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create menu
        self.create_menu()

        # Main paned window
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control panel (left side)
        control_frame = ttk.Frame(paned)
        paned.add(control_frame, weight=1)

        # Create configuration section
        self.create_config_section(control_frame)

        # Graph frame (right side) with scrollbar
        outer_graph_frame = ttk.Frame(paned)
        paned.add(outer_graph_frame, weight=3)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(outer_graph_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create canvas for scrolling
        canvas = tk.Canvas(outer_graph_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbar
        scrollbar.config(command=canvas.yview)
        canvas.config(yscrollcommand=scrollbar.set)

        # Create frame for charts inside canvas
        graph_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=graph_frame, anchor='nw')

        # Create charts
        self.charts = ChartManager(graph_frame)

        # Update scroll region when frame changes
        def _configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        graph_frame.bind('<Configure>', _configure_scroll_region)

    def create_menu(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def create_config_section(self, parent):
        """Create the configuration section"""
        config_frame = ttk.LabelFrame(parent, text="Configuration")
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # Parameters
        param_frame = ttk.Frame(config_frame)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create parameter inputs
        parameters = [
            ('Polling Interval (s):', self.config.polling_interval),
            ('Interrupt Interval (s):', self.config.interrupt_interval),
            ('Simulation Duration (s):', self.config.sim_duration)
        ]

        for i, (label_text, variable) in enumerate(parameters):
            ttk.Label(param_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(param_frame, textvariable=variable, width=10).grid(row=i, column=1, padx=5, pady=5)

        # Mode selection
        ttk.Label(param_frame, text='Mode:').grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        mode_menu = ttk.Combobox(param_frame, textvariable=self.config.mode,
                                 values=['Normal', 'High Load', 'Low Power'], state='readonly')
        mode_menu.grid(row=3, column=1, padx=5, pady=5)

        # Create buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(button_frame, text='Start Simulation', command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text='Reset', command=self.reset_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='Export Results', command=self.export_results).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(config_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)

        # Stats display
        stats_frame = ttk.Frame(config_frame)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_label = ttk.Label(stats_frame, text='Stats: Ready')
        self.stats_label.pack(side=tk.TOP, anchor=tk.W)

        self.latency_label = ttk.Label(stats_frame, text='Avg. Latency: 0.00 ms')
        self.latency_label.pack(side=tk.TOP, anchor=tk.W)

        self.throughput_label = ttk.Label(stats_frame, text='Throughput: 0.00 events/s')
        self.throughput_label.pack(side=tk.TOP, anchor=tk.W)

    def show_about(self):
        """Show the About dialog"""
        about_text = "Interrupt vs Polling Simulation\n\n" \
                     "This application demonstrates the performance differences\n" \
                     "between interrupt-driven and polling-based approaches."
        messagebox.showinfo("About", about_text)

    def export_results(self):
        """Export the simulation results"""
        filename = self.exporter.export_csv(self.engine)
        messagebox.showinfo("Export Complete", f"Results exported to {filename}")

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.engine.reset()
        self.progress['value'] = 0
        self.stats_label.config(text='Stats: Reset')
        self.latency_label.config(text='Avg. Latency: 0.00 ms')
        self.throughput_label.config(text='Throughput: 0.00 events/s')

        # Reset charts
        self.charts.update_events_chart([], [], 0)
        self.charts.update_latency_chart([], [])
        self.charts.update_throughput_chart([])

    def start_simulation(self):
        """Start the simulation"""
        self.start_button.config(state=tk.DISABLED)
        self.reset_simulation()
        self.engine.start(self.update_ui)

    def update_ui(self, current_time, finished=False):
        """Update the UI with current simulation data"""
        if finished:
            self.start_button.config(state=tk.NORMAL)
            return

        # Update progress bar
        progress_value = (current_time / self.config.sim_duration.get()) * 100
        self.progress['value'] = progress_value

        # Update status labels
        self.stats_label.config(
            text=f'Polling: {self.engine.poll_count}, Interrupt: {self.engine.interrupt_count}, '
                 f'CPU Load: {self.engine.cpu_load:.2f}%')

        # Update latency statistics
        avg_poll_latency = np.mean(self.engine.poll_latencies) * 1000 if self.engine.poll_latencies else 0
        avg_interrupt_latency = np.mean(
            self.engine.interrupt_latencies) * 1000 if self.engine.interrupt_latencies else 0

        self.latency_label.config(
            text=f'Avg. Latency: Poll: {avg_poll_latency:.2f} ms, Int: {avg_interrupt_latency:.2f} ms')

        # Calculate current throughput (events per second)
        if current_time > 0:
            throughput = (self.engine.poll_count + self.engine.interrupt_count) / current_time
            self.throughput_label.config(text=f'Throughput: {throughput:.2f} events/s')

        # Update all charts
        self.charts.update_events_chart(
            self.engine.poll_times[:], self.engine.interrupt_times[:], self.engine.cpu_load)
        self.charts.update_latency_chart(
            self.engine.poll_latencies[:], self.engine.interrupt_latencies[:])
        self.charts.update_throughput_chart(self.engine.throughput_data[:])
        
        # Update new charts
        if self.engine.cpu_utilization_data:
            times, usages = zip(*self.engine.cpu_utilization_data)
            self.charts.update_cpu_utilization(times, usages)


if __name__ == '__main__':
    root = tk.Tk()
    app = InterruptPollingSim(root)
    root.mainloop()