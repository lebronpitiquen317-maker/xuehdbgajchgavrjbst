import matplotlib.pyplot as plt
import numpy as np
from .diagnostics import Diagnostics

def plot_estimates(diagnostics: Diagnostics, observed_data: np.ndarray = None, title: str = "Model Estimates"):
    """
    Plot incidence, beta evolution, and ESS.
    """
    times = diagnostics.times
    
    # Get Incidence Quantiles
    inc_qs = diagnostics.get_incidence_quantiles(quantiles=[0.025, 0.5, 0.975])
    
    # Get Beta Quantiles (assuming 'beta' exists)
    beta_qs = diagnostics.get_parameter_quantiles('beta', quantiles=[0.025, 0.5, 0.975])
    
    # Get ESS
    ess = diagnostics.get_ess_trace()
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Incidence
    ax = axes[0]
    if observed_data is not None:
        ax.bar(range(len(observed_data)), observed_data, color='gray', alpha=0.5, label='Observed')
    
    ax.plot(times, inc_qs[:, 1], color='blue', label='Posterior Median')
    ax.fill_between(times, inc_qs[:, 0], inc_qs[:, 2], color='blue', alpha=0.2, label='95% CI')
    ax.set_ylabel('Daily Incidence')
    ax.set_title('Incidence Fit')
    ax.legend()
    
    # 2. Beta
    ax = axes[1]
    if len(beta_qs) > 0:
        ax.plot(times, beta_qs[:, 1], color='green', label='Beta Median')
        ax.fill_between(times, beta_qs[:, 0], beta_qs[:, 2], color='green', alpha=0.2, label='95% CI')
        ax.set_ylabel('Transmission Rate (beta)')
        ax.set_title('Time-varying Transmission')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Beta not found in params', ha='center')
        
    # 3. ESS
    ax = axes[2]
    ax.plot(times, ess, color='black', label='ESS')
    ax.set_ylabel('Effective Sample Size')
    ax.set_xlabel('Time')
    ax.set_title('Particle Filter Performance')
    ax.axhline(y=0, color='r', linestyle='--')
    
    plt.tight_layout()
    return fig

def animate_results(diagnostics: Diagnostics, observed_data: np.ndarray, output_path: str):
    """
    Create an animation of the inference process.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    times = diagnostics.times
    inc_qs = diagnostics.get_incidence_quantiles(quantiles=[0.025, 0.5, 0.975])
    beta_qs = diagnostics.get_parameter_quantiles('beta', quantiles=[0.025, 0.5, 0.975])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Static elements (Observed Data)
    ax1.bar(range(len(observed_data)), observed_data, color='gray', alpha=0.3, label='Observed')
    ax1.set_ylabel('Daily Incidence')
    ax1.set_title('Epidemic Curve Fit (Animated)')
    ax1.legend(loc='upper right')
    
    ax2.set_ylabel('Transmission Rate (Rt/Beta)')
    ax2.set_xlabel('Day')
    ax2.set_ylim(0, 3.0) # Assume beta in reasonable range
    
    # Dynamic elements
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Model Median')
    fill1 = ax1.fill_between([], [], [], color='blue', alpha=0.2)
    
    line2, = ax2.plot([], [], 'g-', linewidth=2, label='Beta Median')
    fill2 = ax2.fill_between([], [], [], color='green', alpha=0.2)
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    
    def update(frame):
        # Frame is current time step index
        # Show history up to frame
        current_times = times[:frame+1]
        
        # Update Incidence
        line1.set_data(current_times, inc_qs[:frame+1, 1])
        # Clear previous fill and redraw
        # ax.collections is a list, we can't call clear() on it directly in newer matplotlib versions if it wraps it?
        # Actually standard list has clear(), but ArtistList might not.
        # Safe way: remove items
        for coll in ax1.collections:
            coll.remove()
        
        # Re-add bar since we cleared collections (bars are collections? No, bars are patches usually, but sometimes collections)
        # plt.bar returns a BarContainer which is patches.
        # fill_between returns a PolyCollection.
        # If we clear collections, we remove fill_between.
        # But we don't want to remove the bars if they are patches.
        # Let's check. 
        # Actually, simpler approach: just remove the specific collection we added last time if we tracked it.
        # But clearing all collections on ax1 is fine IF bars are patches. 
        # Wait, if bars are collections (e.g. if generated differently), they disappear.
        # Let's just try to remove the fill_between specifically.
        
        ax1.fill_between(current_times, inc_qs[:frame+1, 0], inc_qs[:frame+1, 2], color='blue', alpha=0.2)
        
        # Update Beta
        if len(beta_qs) > 0:
            line2.set_data(current_times, beta_qs[:frame+1, 1])
            for coll in ax2.collections:
                coll.remove()
            ax2.fill_between(current_times, beta_qs[:frame+1, 0], beta_qs[:frame+1, 2], color='green', alpha=0.2)
            
        return line1, line2

    anim = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=15)
    anim.save(output_path, writer=writer)
    plt.close(fig)
