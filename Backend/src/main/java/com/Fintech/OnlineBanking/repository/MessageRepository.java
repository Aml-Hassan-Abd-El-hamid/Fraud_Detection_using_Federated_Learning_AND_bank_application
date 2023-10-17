package com.Fintech.OnlineBanking.repository;

import java.util.Optional;




import org.springframework.data.jpa.repository.JpaRepository;

import com.Fintech.OnlineBanking.domain.Message;

public interface MessageRepository extends JpaRepository <Message, Long> {
	Message findByMessageNumber(String message_number);
}
