package com.Fintech.OnlineBanking.repository;

import java.util.List;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import com.Fintech.OnlineBanking.domain.Message;
import com.Fintech.OnlineBanking.domain.PasswordsUser;

public interface PasswordsRepository extends JpaRepository <PasswordsUser, Long>{

	List<PasswordsUser> findAllByUserId(Long id);
}
