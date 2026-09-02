import traceback
from convert_3d import convert_points_to_mesh, export_mesh

square_points = [
    (200, 200),
    (600, 200),
    (600, 600),
    (200, 600),
    (200, 200),  # explicitly closed
]

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800

print("Testing conversion with a clean square...")
print(f"Points: {square_points}")

try:
    mesh = convert_points_to_mesh(
        square_points,
        canvas_width=CANVAS_WIDTH,
        canvas_height=CANVAS_HEIGHT,
        extrude_height=50.0,
    )
    print(f"\nSUCCESS. Mesh type: {type(mesh)}")
    print(f"Is watertight (a proper solid): {mesh.is_watertight}")
    print(f"Vertex count: {len(mesh.vertices)}")
    print(f"Volume: {mesh.volume}")

    out_path = export_mesh(mesh, "test_square.obj")
    print(f"Exported to {out_path}")

    print("\nOpening viewer now...")
    mesh.show()

except Exception:
    print("\n--- FULL ERROR TRACEBACK ---")
    traceback.print_exc()
    print("--- END TRACEBACK ---")
