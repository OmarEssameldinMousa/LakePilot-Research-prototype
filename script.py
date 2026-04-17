import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Team members and their skills
members = {
    'Omar Essam': {
        'Data Engineering': 5,
        'Data Science': 3,
        'Data Analysis': 2,
        'ML Engineering': 3
    },
    'Omar Shrief': {
        'Data Engineering': 1,
        'Data Science': 3,
        'Data Analysis': 4,
        'ML Engineering': 3
    },
    'Omer Karim': {
        'Data Engineering': 0,
        'Data Science': 5,
        'Data Analysis': 4,
        'ML Engineering': 2
    }
}

# Categories (skills)
categories = list(members['Omar Essam'].keys())
N = len(categories)

# Create angles for radar chart
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Complete the loop

# Colors for each member
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Create separate charts for each member
for idx, (member, skills) in enumerate(members.items()):
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.set_facecolor('none')  # Transparent axes background
    
    values = list(skills.values())
    values += values[:1]  # Complete the loop
    
    # Plot the radar
    ax.plot(angles, values, 'o-', linewidth=3, color=colors[idx], markersize=10)
    ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    
    # Set y-axis (skill levels)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=10)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Title
    plt.title(f'{member}\nSkills Profile', size=16, fontweight='bold', y=1.08)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with transparent background
    filename = f'{member.replace(" ", "_").lower()}_skills.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', transparent=True)
    plt.show()
    
    print(f"✅ Saved: {filename}")

print("\n🎉 All radar charts saved successfully!")
