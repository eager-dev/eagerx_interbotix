*************************
eagerx_interbotix package
*************************

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: license

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: codestyle

.. image:: https://github.com/eager-dev/eagerx_interbotix/actions/workflows/ci.yml/badge.svg?branch=master
  :target: https://github.com/eager-dev/eagerx_interbotix/actions/workflows/ci.yml
  :alt: Continuous Integration

.. contents:: Table of Contents
    :depth: 2

What is the *eagerx_interbotix* package?
========================================
This repository interfaces interbotix robots with EAGERx.
EAGERx (Engine Agnostic Gym Environments for Robotics) enables users to easily define new tasks, switch from one sensor to another, and switch from simulation to reality with a single line of code by being invariant to the physics engine.
`The core repository is available here <https://github.com/eager-dev/eagerx>`_.
`Full documentation and tutorials (including package creation and contributing) are available here <https://eagerx.readthedocs.io/en/master/>`_.

Installation
============

You can install the package using pip:

.. code:: shell

    pip3 install eagerx-interbotix

.. note::
    EAGERx depends on a minimal ROS installation. Fortunately, you **can** use eagerx anywhere as you would any python package,
    so it does **not** impose a ROS package structure on your project.

Dependencies (reality only)
===========================

We require the interbotix ros package to control the interbotix manipulators in the real world.

If you do not have rosdep already installed (check if your shell recognizes the command ``rosdep -help``), run the code below.
If you use the melodic distribution, use ``python-rosdep`` instead.

.. code:: shell

    sudo apt-get install python3-rosdep
    sudo rosdep init
    rosdep update

Reboot your computer if you had to install rosdep.

Then install the package by typing:

.. code:: shell

        cd ~
        curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
        chmod +x xsarm_amd64_install.sh
        ./xsarm_amd64_install.sh

If a previous installation attempt failed, remove directory ``~/interbotix_ws``, before retrying.

Real-world experiments
======================
After you installed the interbotix ros package, you should launch the driver.
Open a terminal, source ``interbotix_ws/devel/setup.bash``, and run the command below.

- Replace ``$ROBOT_TYPE`` with the robot model (e.g. vx300s, px150, etc..).

- Replace ``$OBJ_NAME`` with the name you specified when creating the arm object spec in ``eagerx``.

- Optionally, you can set ``use_sim:=True`` to use a mock arm in ``RViz``.

- See `here <https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_control/launch/xsarm_control.launch>`_ for more options.

.. code:: shell

    roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=$ROBOT_TYPE robot_name:=$OBJ_NAME use_sim:=False

Cite EAGERx
===========
If you are using EAGERx for your scientific publications, please cite:

.. code:: bibtex

    @article{eagerx,
        author  = {van der Heijden, Bas and Luijkx, Jelle, and Ferranti, Laura and Kober, Jens and Babuska, Robert},
        title = {EAGERx: Engine Agnostic Gym Environment for Robotics},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/eager-dev/eagerx}}
    }

Acknowledgements
================
EAGERx is funded by the `OpenDR <https://opendr.eu/>`_ Horizon 2020 project.
