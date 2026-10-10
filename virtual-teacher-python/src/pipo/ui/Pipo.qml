import QtQuick
import QtQuick.Shapes

// Pipo, drawn and animated in QML: breathes, blinks, tilts when listening, looks up and spins thoughts
// when thinking, and opens his mouth with the loudness of his own voice. Coordinates are a 300x300 canvas
// scaled to the item's size, like the SVG in the web app.
Item {
    id: root
    property string phase: "starting"
    property real mouth: 0
    readonly property real s: width / 300

    readonly property bool speaking: phase === "speaking"
    readonly property bool listening: phase === "listening"
    readonly property bool thinking: phase === "thinking"

    // Glow behind him, stronger while he listens or speaks.
    Rectangle {
        x: 150 * root.s - width / 2; y: 150 * root.s - height / 2
        width: 250 * root.s; height: width; radius: width / 2
        color: root.listening ? win.accent : win.amber
        opacity: root.listening ? 0.16 : (root.speaking ? 0.06 + root.mouth * 0.14 : 0.04)
        Behavior on opacity { NumberAnimation { duration: 200 } }
        Behavior on color { ColorAnimation { duration: 300 } }
    }

    Item {
        id: head
        width: parent.width; height: parent.height
        transformOrigin: Item.Bottom
        rotation: root.listening ? -6 : (root.thinking ? 4 : 0)
        scale: root.listening ? 1.03 : 1
        Behavior on rotation { SpringAnimation { spring: 2.5; damping: 0.18 } }
        Behavior on scale { SpringAnimation { spring: 2.5; damping: 0.2 } }

        // Breathing.
        transform: Translate {
            SequentialAnimation on y {
                loops: Animation.Infinite
                NumberAnimation { to: -5 * root.s; duration: 2000; easing.type: Easing.InOutSine }
                NumberAnimation { to: 0; duration: 2000; easing.type: Easing.InOutSine }
            }
        }

        // Antenna.
        Rectangle { x: 147 * root.s; y: 28 * root.s; width: 6 * root.s; height: 32 * root.s; radius: 3 * root.s; color: "#5f7fa8" }
        Rectangle {
            id: bulb
            x: 140 * root.s; y: 14 * root.s; width: 20 * root.s; height: width; radius: width / 2
            color: root.phase === "error" ? win.red : (root.listening ? win.accent : win.amber)
            Behavior on color { ColorAnimation { duration: 300 } }
            SequentialAnimation on opacity {
                running: root.listening || root.thinking
                loops: Animation.Infinite
                alwaysRunToEnd: true
                NumberAnimation { to: 0.35; duration: root.thinking ? 250 : 500 }
                NumberAnimation { to: 1; duration: root.thinking ? 250 : 500 }
            }
        }

        // Thoughts orbiting the antenna while he thinks.
        Repeater {
            model: 3
            Rectangle {
                required property int index
                width: 9 * root.s; height: width; radius: width / 2
                color: win.accent
                opacity: root.thinking ? 0.8 : 0
                Behavior on opacity { NumberAnimation { duration: 250 } }
                property real angle: index * 120
                x: 150 * root.s + Math.cos(angle * Math.PI / 180) * 34 * root.s - width / 2
                y: 24 * root.s + Math.sin(angle * Math.PI / 180) * 14 * root.s - height / 2
                NumberAnimation on angle { running: root.thinking; from: index * 120; to: index * 120 + 360; duration: 1400; loops: Animation.Infinite }
            }
        }

        // Body.
        Rectangle {
            x: 55 * root.s; y: 55 * root.s; width: 190 * root.s; height: 190 * root.s; radius: 80 * root.s
            gradient: Gradient {
                GradientStop { position: 0; color: "#9fb7d9" }
                GradientStop { position: 1; color: "#5f7fa8" }
            }
        }
        // Cheeks.
        Rectangle { x: 80 * root.s; y: 181 * root.s; width: 32 * root.s; height: 18 * root.s; radius: height / 2; color: "#e7b4a8"; opacity: root.speaking ? 0.75 : 0.55 }
        Rectangle { x: 188 * root.s; y: 181 * root.s; width: 32 * root.s; height: 18 * root.s; radius: height / 2; color: "#e7b4a8"; opacity: root.speaking ? 0.75 : 0.55 }

        // Eyes: look up when thinking, blink every few seconds.
        Item {
            id: eyes
            width: parent.width; height: parent.height
            y: (root.thinking ? -6 : 0) * root.s
            Behavior on y { NumberAnimation { duration: 300; easing.type: Easing.OutBack } }
            property real lid: 1
            SequentialAnimation on lid {
                loops: Animation.Infinite
                PauseAnimation { duration: 3800 }
                NumberAnimation { to: 0.1; duration: 90 }
                NumberAnimation { to: 1; duration: 120 }
                PauseAnimation { duration: 900 }
            }
            Repeater {
                model: [112, 188]
                Item {
                    required property int modelData
                    x: (modelData - 17) * root.s; y: 119 * root.s; width: 34 * root.s; height: 42 * root.s
                    transform: Scale { origin.y: 21 * root.s; yScale: eyes.lid }
                    Rectangle { anchors.fill: parent; radius: width / 2; color: "#f4f1ea" }
                    Rectangle { x: 11 * root.s; y: 15 * root.s; width: 18 * root.s; height: width; radius: width / 2; color: "#232323" }
                    Rectangle { x: 18 * root.s; y: 17 * root.s; width: 6 * root.s; height: width; radius: width / 2; color: "white" }
                }
            }
        }

        // Mouth: an opening that follows his voice, or a smile when quiet.
        Rectangle {
            id: mouthOpen
            readonly property real open: 4 + root.mouth * 26
            width: (22 - root.mouth * 4) * 2 * root.s
            height: open * root.s
            radius: Math.min(width, height) / 2
            x: 150 * root.s - width / 2
            y: (196 + open / 4) * root.s - height / 2
            color: "#3b2a2a"
            visible: root.speaking
        }
        Shape {
            visible: !root.speaking
            width: parent.width; height: parent.height
            preferredRendererType: Shape.CurveRenderer
            ShapePath {
                strokeColor: "#3b2a2a"; strokeWidth: 5 * root.s; fillColor: "transparent"; capStyle: ShapePath.RoundCap
                startX: 130 * root.s; startY: 194 * root.s
                PathQuad { controlX: 150 * root.s; controlY: 206 * root.s; x: 170 * root.s; y: 194 * root.s }
            }
        }
    }
}
