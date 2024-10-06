import streamlit as st
import obspy
import numpy as np
import pandas as pd
import tempfile
import plotly.graph_objs as go
from scipy.io.wavfile import write
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import base64
import joblib  
from sklearn.preprocessing import StandardScaler  
import os

def create_date_columns(df):
    df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] = pd.to_datetime(df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
    
    # extracting year, month, day, hour, minute, and second from the datetime column
    df['year'] = df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].dt.year
    df['month'] = df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].dt.month
    df['day'] = df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].dt.day
    df['hour'] = df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].dt.hour
    df['minute'] = df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].dt.minute
    df['second'] = df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].dt.second
    
    df = df.drop(columns=['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], errors='ignore')
    
    return df


def plot_on_off_triggers(cft, tr_times, tr_data, thr_on=4, thr_off=1.5):
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))
    st.write("Trigger On/Off Indices:", on_off)

    if len(on_off) == 0:
        st.warning("No triggers detected with the specified thresholds.")
        return None  # Return None if no triggers detected

    # Plot on and off triggers
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    unique_on_off = np.unique(on_off, axis=0)  # Get unique trigger pairs

    for i, trigger in enumerate(unique_on_off):
        if trigger[0] < len(tr_times) and trigger[1] < len(tr_times):
            ax.axvline(x=tr_times[trigger[0]], color='red', linestyle='--', label='Trig. On' if i == 0 else "")
            ax.axvline(x=tr_times[trigger[1]], color='purple', linestyle='--', label='Trig. Off' if i == 0 else "")

    # Plot seismogram
    ax.plot(tr_times, tr_data)
    ax.set_xlim([min(tr_times), max(tr_times)])
    ax.legend()

    return fig  # Return the figure for plotting


def read_miniseed_file(uploaded_file):
    st.write("Reading the MiniSEED file...")
    return obspy.read(uploaded_file)

def process_trace_data(stream):
    df_list = []
    st.write("## Trace Metadata")
    for trace in stream:
        st.write(trace.stats)
        start_time = trace.stats.starttime
        time_array = np.array([start_time + i / trace.stats.sampling_rate for i in range(len(trace.data))])
        velocity = trace.data * trace.stats.sampling_rate

        df = pd.DataFrame({
            "time_abs(%Y-%m-%dT%H:%M:%S.%f)": time_array.astype(str),
            "time_rel(sec)": np.arange(len(trace.data)) / trace.stats.sampling_rate,
            "velocity(m/s)": velocity
        })
        
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

def save_to_csv(final_df):
    csv_file = "seismic_data.csv"
    final_df.to_csv(csv_file, index=False)
    st.success(f"Data successfully saved as: **{csv_file}**")

def plot_waveform(stream):
    st.write("## Interactive Waveform Plot")
    fig = go.Figure()

    for i, trace in enumerate(stream):
        time_array = np.arange(len(trace.data)) / trace.stats.sampling_rate
        fig.add_trace(go.Scatter(
            x=time_array,
            y=trace.data,
            mode='lines',
            name=f'Trace {i+1}',
            line=dict(width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title="Waveform Plot",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        legend_title="Traces",
        hovermode='x unified',
        template='plotly',
        width=800,
        height=400
    )

    st.plotly_chart(fig)

def play_audio(stream):
    st.write("## Hear the Seismic Data")
    trace = stream[0]
    data = trace.data
    data_normalized = data / np.max(np.abs(data))
    sample_rate = 44100

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
        write(tmp_wav_file.name, sample_rate, data_normalized.astype(np.float32))
        st.audio(tmp_wav_file.name)

def filter_trace(st, minfreq=0.5, maxfreq=1.0):
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data
    f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)
    return f, t, sxx, tr_times_filt, tr_data_filt

def sta_lta(st):
    tr = st.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data
    df = tr.stats.sampling_rate
    sta_len = 120
    lta_len = 600
    cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))
    return cft, tr_times, tr_data

def spectrogram_plot(f, t, sxx, tr_times_filt, tr_data_filt, arrival=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(tr_times_filt, tr_data_filt)
    if arrival is not None:
        ax[0].axvline(x=arrival, color='red', label='Detection')
    ax[0].set_xlim([min(tr_times_filt), max(tr_times_filt)])
    ax[0].set_ylabel('Velocity (m/s)')
    ax[0].set_xlabel('Time (s)')
    ax[0].legend(loc='upper left')
    
    vals = ax[1].pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    ax[1].set_xlim([min(tr_times_filt), max(tr_times_filt)])
    ax[1].set_xlabel('Time (Day Hour:Minute)', fontweight='bold')
    ax[1].set_ylabel('Frequency (Hz)', fontweight='bold')
    if arrival is not None:
        ax[1].axvline(x=arrival, color='red')
    cbar = plt.colorbar(vals, ax=ax[1], orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

    st.pyplot(fig)

def plot_char_func(tr_times, cft):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(tr_times, cft)
    ax.set_xlim([min(tr_times), max(tr_times)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Characteristic function')


@st.cache_resource
def load_scaler(filename):
    return joblib.load(filename)

def load_model(filename):
    return joblib.load(filename)

loaded_model = load_model('./models/best_model.joblib')
loaded_scaler = load_scaler('./models/scaler.joblib')

def predict_new_data(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    return predictions


################################# MAIN ############################################
from streamlit.components.v1 import html

# HTML and JavaScript for starry night background
starry_night_bg = """
<style>
body {
    padding: 0;
    margin: 0;
    height: 100vh;
    background: linear-gradient(0deg, rgba(13, 35, 93, 1) 0%, rgba(0, 5, 8, 1) 50%); /* Changed from 70% to 50% */
    overflow: hidden;  /* Prevent scrollbars */
}
#canvas {
    position: fixed;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    z-index: -1; /* Send the canvas to the back */
}
</style>
<canvas id="canvas"></canvas>
<script>
let canvas, ctx, w, h, moon, stars = [], meteors = [];

function init() {
    canvas = document.querySelector("#canvas");
    ctx = canvas.getContext("2d");
    resizeReset();
    moon = new Moon();
    for (let a = 0; a < w * h * 0.0002; a++) {  // Increased density of stars
        stars.push(new Star());
    }
    for (let b = 0; b < 5; b++) {  // Increased number of meteors
        meteors.push(new Meteor());
    }
    animationLoop();
}

function resizeReset() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
}

function animationLoop() {
    ctx.clearRect(0, 0, w, h);
    drawScene();
    requestAnimationFrame(animationLoop);
}

function drawScene() {
    moon.draw();
    stars.map((star) => {
        star.update();
        star.draw();
    });
    meteors.map((meteor) => {
        meteor.update();
        meteor.draw();
    });
}

class Moon {
    constructor() {
        this.x = 90;
        this.y = 90;
        this.size = 90;  // Increased moon size
    }
    draw() {
        ctx.save();
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.shadowColor = "rgba(254, 247, 144, .7)";
        ctx.shadowBlur = 70;
        ctx.fillStyle = "rgba(254, 247, 144, 1)";
        ctx.fill();
        ctx.closePath();
        ctx.restore();
    }
}

class Star {
    constructor() {
        this.x = Math.random() * w;
        this.y = Math.random() * h;
        this.size = Math.random() * 4 + 2;  // Increased star size
        this.speed = Math.random() * 0.05 + 0.01;
    }
    draw() {
        ctx.fillStyle = "rgba(255, 255, 255, 1)";
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
    update() {
        this.x += this.speed;
        if (this.x > w) {
            this.x = 0;
        }
    }
}

class Meteor {
    constructor() {
        this.x = Math.random() * w;
        this.y = Math.random() * h;
        this.length = Math.random() * 50 + 20; // Decreased length range for shorter meteors
        this.speed = Math.random() * 2 + 2; 
    }
    draw() {
        ctx.strokeStyle = "rgba(255, 255, 255, 1)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(this.x - this.length, this.y + this.length);
        ctx.stroke();
    }
    update() {
        this.x -= this.speed;
        if (this.x < -this.length) {
            this.x = Math.random() * w;
            this.y = Math.random() * h;
        }
    }
}

window.addEventListener("resize", resizeReset);
window.onload = init;
</script>
"""

# Main section of the Streamlit app
#st.set_page_config(layout="wide")

st.markdown(starry_night_bg, unsafe_allow_html=True)
# HTML and CSS for background image
background_image = """
<style>
body {
    padding: 0;
    margin: 0;
    height: 100vh;
    background: url('./pic1.jpg') no-repeat center center fixed; /* Replace with your image URL */
    background-size: cover; /* Cover the entire background */
}
</style>
"""

# Main section of the Streamlit app
st.markdown(background_image, unsafe_allow_html=True)

#-----------------------------------------SET BACKGROUND IMAGE--------------------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def cooling_highlight(val):
    color = '#ACE5EE' if val else '#F2F7FA'
    return f'background-color: {color}'

#-----------------------------------------PAGE CONFIGURATIONS--------------------------------------
#st.set_page_config(
#        page_title="Main page",
#)


#define CSS styles for the sidebar
sidebar_styles = """
    .sidebar-content {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 1rem;
        
    }
    .sidebar-image {
        max-width: 150px;
        display: block;
        margin: 0 auto;
    }
"""

def set_background_color(hex_color, color):
    style = f"""
        <style>
        .background-text {{
            background-color: {hex_color};
            padding: 5px; /* Adjust padding as needed */
            border-radius: 5px; /* Rounded corners */
            color: {color}; /* Text color */
        }}
        </style>
    """
    return style


if __name__ == "__main__":
    # Add the background animation
    set_background('./pic1.jpg')
    html(starry_night_bg)
    
    def header(url): 
        st.markdown(f'''
        <p style="color:white;
                background: linear-gradient(90deg, #0d235d, #000508); 
                font-size:30px;
                padding:10px;
                border-radius:5px;">
            {url}
        </p>
        ''', unsafe_allow_html=True)

    def header2(url): 
        st.markdown(f'''
        <p style="color:white;
                background: linear-gradient(90deg, #0d235d, #000508); 
                font-size:22px;
                padding:10px;
                display: inline-block;
                border-radius:5px;">
            {url}
        </p>
        ''', unsafe_allow_html=True)

    #st.title("Seismic Detection across the Solar System!")
    header('Seismic Detection across the Solar System!')
    

    uploaded_file = st.file_uploader("Upload an .mseed file", type=["mseed"])

    if uploaded_file:
        stream = read_miniseed_file(uploaded_file)
        
        st.write('## File Information')
        st.write(stream)

        final_df = process_trace_data(stream)

        st.session_state=final_df
        results_folder = './results'  
              
        results_file_path = os.path.join(results_folder, 'final_df.csv')
        final_df.to_csv(results_file_path, index=False)
        st.success(f"The dataframe is saved as: **{results_file_path}**")
   
        #save_to_csv(final_df)
        #final_df= st.dataframe(final_df)
        plot_waveform(stream)
        play_audio(stream)

        header2("Let's filter the trace and apply STA/LTA algorithm to detect seismic events!")
        header2("The spectrogram below will show us the exact time where a seismic event happened.")
        min_freq = st.number_input("Minimum Frequency (Hz)", min_value=0.0, max_value=10.0, value=0.5)
        max_freq = st.number_input("Maximum Frequency (Hz)", min_value=0.0, max_value=20.0, value=1.0)

        if st.button("Apply Filter and STA/LTA"):
            f, t, sxx, tr_times_filt, tr_data_filt = filter_trace(stream, min_freq, max_freq)
            cft, tr_times, tr_data = sta_lta(stream)

            spectrogram_plot(f, t, sxx, tr_times_filt, tr_data_filt)
            plot_char_func(tr_times, cft)
            header2("Now let's check up the beginning and the ending of different types of seismic events!")
            thr_on = st.number_input("Trigger On Threshold", value=4.0)
            thr_off = st.number_input("Trigger Off Threshold", value=1.5)
            
            fig = plot_on_off_triggers(cft, tr_times, tr_data, thr_on, thr_off)  # Get the figure
    
            if fig is not None:  # Check if the figure is valid before plotting
                st.pyplot(fig) 
        print("done")
        if st.button("Detect Seismic Events!"):

            #new_data_example = st.session_state 

            #example data due to low computation on laptop
            #data from catalogue lunar GradA
            data_dict = {
                "time_abs(%Y-%m-%dT%H:%M:%S.%f)": [
                    "1970-01-19T20:25:00.740472Z",
                    "1970-01-19T20:25:00.891415Z",
                    "1970-01-19T20:25:01.042358Z",
                    "1970-01-19T20:25:01.193302Z",
                ],
                "time_rel(sec)": [
                    73500,  # Updated to 73500 for each entry
                    73500,
                    73500,
                    73500,
                ],
                "velocity(m/s)": [
                    9.004942e-10,
                    4.575971e-10,
                    -4.284551e-10,
                    -1.209429e-09,
                ],
            }
            new_data_example = pd.DataFrame(data_dict)
            new_data_example = create_date_columns(new_data_example)

            # Create a list to hold predictions
            predictions_list = []

            # Iterate over each row and predict
            for index, row in new_data_example.iterrows():
                
                features = row.values.reshape(1, -1)  # Reshape for a single sample prediction
                prediction = predict_new_data(loaded_model, loaded_scaler, features)
                predictions_list.append(prediction[0])  

            # Add the predictions as a new column
            new_data_example['Predictions'] = predictions_list

            # Show the DataFrame with predictions
            st.write("## Results:")
            st.dataframe(new_data_example)

            results_folder = './results'  
              
            results_file_path = os.path.join(results_folder, 'predictions_with_results.csv')
            new_data_example.to_csv(results_file_path, index=False)
            st.success(f"Predictions saved as: **{results_file_path}**")
